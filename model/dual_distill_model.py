from typing import *

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import transformers
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch import optim, nn
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

from ._loss import LossCalculator
from ._metrics import cal_flop, cal_speed
from ._utils import teacher_load
from .component.clip_model import CLIPModel
from .component.output import CLIPOutput


class DualDistillModel(pl.LightningModule):
    def __init__(self, image_student: nn.Module, text_student: nn.Module, teacher_need_layers: List, teacher_name: str,
                 loss_control_para: Dict, warm_steps, total_steps, weight_decay, lr: float,
                 download_root: str, map_type: Optional[str] = None, init_type: Optional[str] = None,
                 norm=False):
        super().__init__()
        self.save_hyperparameters(ignore=['image_student', 'text_student'])

        # define model
        self.student = CLIPModel(True, image_student, text_student, norm)
        self.teacher_name = teacher_name
        self.teacher = teacher_load(teacher_name, download_root, 'all', need_layers=teacher_need_layers)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output()
        # define metric
        self.k_list = [i for i in [1, 2, 3, 4, 5, 10]]
        self.acc_metrics = []
        for k in self.k_list:
            self.acc_metrics.append(Accuracy(top_k=k))
        self.acc_metrics = nn.ModuleList(self.acc_metrics)

    def on_train_start(self):
        if self.global_rank == 0:
            if isinstance(self.logger, WandbLogger):
                self.logger.log_hyperparams({'student_para': self.student.hyper_para()})
                wandb.save('./*.py')
                wandb.save('./data/*.py')
                wandb.save('./model/*.py')
                wandb.save('./model/component/*.py')
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.log_hyperparams(self.hparams, {"hp/stu_acc_top1": 0, "hp/stu_acc_top10": 0})
        dummy_input = (
            torch.randint(high=49407, size=(1, 77), device=self.device),
            torch.rand(size=(1, 3, 224, 224), device=self.device))
        self.speed_test(self.student, dummy_input, pre_fix='stu_')
        self.speed_test(self.teacher, dummy_input, pre_fix='tea_')

    def speed_test(self, model, dummy_input, pre_fix='stu_'):
        flops, param = cal_flop(model, dummy_input)
        mean_syn, std_syn, mean_fps = cal_speed(model, dummy_input)
        metric_dict = {
            pre_fix + 'flops': flops,
            pre_fix + 'param': param,
            pre_fix + 'mean_times': mean_syn,
            pre_fix + 'std_times': std_syn,
            pre_fix + 'mean_fps': mean_fps
        }
        self.log_dict(metric_dict, sync_dist=True)

    def forward(self, inputs) -> Tuple[CLIPOutput, CLIPOutput]:
        image, text = inputs
        student_outs: CLIPOutput = self.student(text, image, self.need_return_para)
        teacher_outs: CLIPOutput = self.teacher(text, image, self.need_return_para)
        return student_outs, teacher_outs

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        self.log_info('train', loss, cal_res, batch_size=len(inputs))
        return loss

    def validation_step(self, batch, batch_idx):

        student_outs, teacher_outs = self.forward(batch)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        self.log_info('val', loss, cal_res, batch_size=len(batch))
        return {
            'stu_image_outs': self.all_gather(student_outs.visual_output.last_representation),
            'stu_text_outs': self.all_gather(student_outs.text_output.last_representation),
            'tea_image_outs': self.all_gather(teacher_outs.visual_output.last_representation),
            'tea_text_outs': self.all_gather(teacher_outs.text_output.last_representation),
        }

    def validation_step_end(self, step_out):
        return step_out

    def validation_epoch_end(self, outputs):
        # [gpu_num, batch, batch]
        stu_image_outs = []
        stu_text_outs = []
        tea_image_outs = []
        tea_text_outs = []

        for batch in outputs:
            stu_image_out, stu_text_out = batch['stu_image_outs'], batch['stu_text_outs']
            tea_image_out, tea_text_out = batch['tea_image_outs'], batch['tea_text_outs']
            embedding = stu_image_out.shape[-1]
            stu_image_outs.append(stu_image_out.reshape(-1, embedding))
            stu_text_outs.append(stu_text_out.reshape(-1, embedding))
            tea_image_outs.append(tea_image_out.reshape(-1, embedding))
            tea_text_outs.append(tea_text_out.reshape(-1, embedding))

        stu_image_outs = torch.cat(stu_image_outs, dim=0).float()
        stu_text_outs = torch.cat(stu_text_outs, dim=0).float()
        tea_image_outs = torch.cat(tea_image_outs, dim=0).float()
        tea_text_outs = torch.cat(tea_text_outs, dim=0).float()

        stu_logits, _ = norm_and_logits(stu_image_outs, stu_text_outs)
        tea_logits, _ = norm_and_logits(tea_image_outs, tea_text_outs)
        label = torch.arange(stu_logits.shape[0], device=self.device)
        tea_softmax_logits = nn.functional.softmax(tea_logits, dim=1)
        stu_softmax_logits = nn.functional.softmax(stu_logits, dim=1)

        softmax_mean_score = torch.diagonal(stu_softmax_logits).mean()
        mean_score = torch.diagonal(stu_logits).mean()

        self.log('softmax_mean_score', softmax_mean_score, batch_size=stu_logits.shape[0], sync_dist=True)
        self.log('mean_score', mean_score, batch_size=stu_logits.shape[0], sync_dist=True)
        self.logger.log_image(key="stu_logtis_map", images=[plt.imshow(stu_logits.cpu())])
        self.logger.log_image(key="stu_softmax_logtis_map", images=[plt.imshow(stu_softmax_logits.cpu())])
        if self.current_epoch == 0:
            tea_softmax_mean_score = torch.diagonal(tea_softmax_logits).mean()
            tea_mean_score = torch.diagonal(tea_logits).mean()
            self.log('tea_softmax_mean_score', tea_softmax_mean_score, batch_size=stu_logits.shape[0],
                     sync_dist=True)
            self.log('tea_mean_score', tea_mean_score, batch_size=stu_logits.shape[0], sync_dist=True)
            self.logger.log_image(key="tea_logtis_map", images=[plt.imshow(tea_logits.cpu())])
            self.logger.log_image(key="tea_softmax_logtis_map", images=[plt.imshow(tea_softmax_logits.cpu())])

        for i, metric in enumerate(self.acc_metrics):
            metric(stu_logits, label)
            self.log('hp_metric/stu_acc_top{}'.format(self.k_list[i]), metric, metric_attribute='acc_metrics',
                     batch_size=stu_logits.shape[0], sync_dist=True)
            if self.current_epoch == 0:
                acc_tea = accuracy(tea_logits, label, top_k=self.k_list[i])
                self.log('hp_metric/tea_acc_top{}'.format(self.k_list[i]), acc_tea, batch_size=tea_logits.shape[0],
                         sync_dist=True)

    def log_info(self, stage, loss, cal_res, batch_size):

        self.log("{}/loss".format(stage), loss, batch_size=batch_size, sync_dist=True)
        for loss_name, loss_res in cal_res.items():
            self.log("{}/{}".format(stage, loss_name), loss_res, batch_size=batch_size, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warm_steps,
            num_training_steps=self.hparams.total_steps
        )
        return [optimizer], [scheduler]


def norm_and_logits(img_encode, text_encode):
    img_encode = img_encode / img_encode.norm(dim=1, keepdim=True)
    text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
    logits = img_encode @ text_encode.t()
    return logits, logits.T
