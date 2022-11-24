from typing import *

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import transformers
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import optim, nn
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

    def on_train_start(self):
        self.logger_begin()
        dummy_input = (
            torch.randint(high=49407, size=(1, 77), device=self.device),
            torch.rand(size=(1, 3, 224, 224), device=self.device))
        self.speed_test(self.student, dummy_input, prefix='stu')
        self.speed_test(self.teacher, dummy_input, prefix='tea')

    @rank_zero_only
    def logger_begin(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_hyperparams({'student_para': self.student.hyper_para()})
            wandb.save('./*.py')
            wandb.save('./data/*.py')
            wandb.save('./model/*.py')
            wandb.save('./model/component/*.py')
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(self.hparams, {"hp/stu_acc_top1": 0, "hp/stu_acc_top10": 0})

    @rank_zero_only
    def speed_test(self, model, dummy_input, prefix):
        with torch.no_grad():
            flops, param = cal_flop(model, dummy_input)
            mean_syn, std_syn, mean_fps = cal_speed(model, dummy_input)
        metric_dict = {
            f'{prefix}_flops': flops,
            f'{prefix}_param': param,
            f'{prefix}_mean_times': mean_syn,
            f'{prefix}_std_times': std_syn,
            f'{prefix}_mean_fps': mean_fps
        }
        self.log_dict(metric_dict, rank_zero_only=True, sync_dist=False)

    def forward(self, inputs) -> Tuple[CLIPOutput, CLIPOutput]:
        image, text = inputs
        student_outs: CLIPOutput = self.student(text, image, self.need_return_para)
        teacher_outs: CLIPOutput = self.teacher(text, image, self.need_return_para)
        if self.hparams.norm:
            student_outs, teacher_outs = norm_last_representation(student_outs, teacher_outs)
        return student_outs, teacher_outs

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)

        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        self.log_info('train', loss, cal_res, batch_size=len(inputs))

        stu_logits, _ = norm_and_logits(student_outs.visual_output.last_representation,
                                        student_outs.text_output.last_representation)
        tea_logits, _ = norm_and_logits(teacher_outs.visual_output.last_representation,
                                        teacher_outs.text_output.last_representation)
        self.log_acc(stu_logits, stage='train', prefix='stu')
        self.log_acc(tea_logits, stage='train', prefix='tea')

        if self.global_step % 2000 == 0:
            self.log_heatmap(stu_logits, stage='train', prefix='stu')

        return loss

    def validation_step(self, batch, batch_idx):

        student_outs, teacher_outs = self.forward(batch)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        self.log_info('val', loss, cal_res, batch_size=len(batch))
        stu_logits, _ = norm_and_logits(student_outs.visual_output.last_representation,
                                        student_outs.text_output.last_representation)
        tea_logits, _ = norm_and_logits(teacher_outs.visual_output.last_representation,
                                        teacher_outs.text_output.last_representation)
        self.log_acc(stu_logits, 'val_step', prefix='stu')
        self.log_acc(tea_logits, stage='val_step', prefix='tea')
        if self.global_step % 2000 == 0:
            self.log_heatmap(stu_logits, stage='val_step', prefix='stu')
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

        stu_image_tea_text_logits, _ = norm_and_logits(stu_image_outs, tea_text_outs)
        stu_text_tea_image_logits, _ = norm_and_logits(tea_image_outs, stu_text_outs)

        self.log_acc(stu_logits, stage='val', prefix='stu')
        self.log_acc(stu_image_tea_text_logits, stage='val', prefix='stu_image_tea_text')
        self.log_acc(stu_text_tea_image_logits, stage='val', prefix='stu_text_tea_image')

        if self.current_epoch % 50 == 0:
            self.log_heatmap(stu_logits, stage='val', prefix='stu')

        if self.current_epoch == 0:
            self.log_heatmap(tea_logits, stage='val', prefix='tea')
            self.log_acc(tea_logits, stage='val', prefix='tea')

        return

    def log_info(self, stage, loss, cal_res, batch_size):

        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)
        for loss_name, loss_res in cal_res.items():
            self.log(f"{stage}/{loss_name}", loss_res, batch_size=batch_size, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warm_steps,
            num_training_steps=self.hparams.total_steps
        )
        return [optimizer], [scheduler]

    def log_heatmap(self, logits, stage, prefix):
        softmax_logits = nn.functional.softmax(logits, dim=1)
        softmax_mean_score = torch.diagonal(softmax_logits).mean()
        mean_score = torch.diagonal(logits).mean()

        self.log(f'{stage}_{prefix}_softmax_mean_score', softmax_mean_score, batch_size=logits.shape[0], sync_dist=True)
        self.log(f'{stage}_{prefix}_mean_score', mean_score, batch_size=logits.shape[0], sync_dist=True)
        if self.global_rank == 0:
            self.logger.log_image(key=f"{stage}_{prefix}_logits",
                                  images=[plt.imshow(logits.detach().cpu()),
                                          plt.imshow(softmax_logits.detach().cpu())],
                                  caption=[f"{stage}_{prefix}_map", f"{stage}_{prefix}_softmax_map"])

    def log_acc(self, logits, stage, prefix):
        label = torch.arange(logits.shape[0], device=self.device)
        for k in self.k_list:
            acc = accuracy(logits, label, top_k=k)
            if stage == 'val':
                self.log(f'hp_metric/{prefix}_acc_top{k}', acc, batch_size=logits.shape[0], sync_dist=True)
            else:
                self.log(f'{stage}/{stage}_{prefix}_acc_top{k}', acc, batch_size=logits.shape[0], sync_dist=True)


def norm_and_logits(img_encode, text_encode):
    img_encode = img_encode / img_encode.norm(dim=1, keepdim=True)
    text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
    logits = img_encode @ text_encode.t()
    return logits, logits.T


def norm_last_representation(stu_outs, tea_outs):
    stu_outs.visual_output.last_representation /= stu_outs.visual_output.last_representation.norm(dim=-1, keepdim=True)
    stu_outs.text_output.last_representation /= stu_outs.text_output.last_representation.norm(dim=-1, keepdim=True)
    tea_outs.visual_output.last_representation /= tea_outs.visual_output.last_representation.norm(dim=-1, keepdim=True)
    tea_outs.text_output.last_representation /= tea_outs.text_output.last_representation.norm(dim=-1, keepdim=True)

    return stu_outs, tea_outs
