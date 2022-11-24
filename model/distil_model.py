from typing import *

import pytorch_lightning as pl
import torch
import transformers
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import optim, nn
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

from ._loss import LossCalculator
from ._metrics import cal_speed, cal_flop
from ._utils import teacher_load


class DistillModel(pl.LightningModule):
    def __init__(self, student_encoder: torch.nn.Module,
                 teacher_name: str, loss_control_para: Dict, download_root: str, norm: bool,
                 teacher_need_layers: List, model_type: str = 'text', map_type: str = 'mid', init_type=None,
                 warm_steps=10, total_steps=200, weight_decay=1e-3, lr: float = 1e-3):
        super().__init__()
        if model_type not in ['text', 'image']:
            raise ValueError(f"the model_type should in ['text', 'image'], bug got {model_type}")
        self.save_hyperparameters(ignore=['student_encoder'])

        # 定义模型
        self.student = student_encoder
        self.teacher_name = teacher_name
        self.teacher = teacher_load(teacher_name, download_root, model_type,
                                    need_layers=teacher_need_layers)
        if len(self.teacher.need_layers) != len(self.student.need_layers):
            raise ValueError(
                f'the teacher need_layers length is not equal to student need_layers length. '
                f'But get teacher: {self.teacher.need_layers}, student: {self.student.need_layers}')
        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 定义指标
        self.k_list = [i for i in [1, 2, 3, 4, 5, 10]]
        self.acc_metrics = torch.nn.ModuleList()
        for k in self.k_list:
            self.acc_metrics.append(Accuracy(top_k=k))

    def on_train_start(self):
        self.logger_begin()
        if self.hparams.model_type == 'image':
            dummy_input = torch.rand(size=(1, 3, 224, 224), device=self.device)
        else:
            dummy_input = torch.rand(size=(1, 77), device=self.device)
        self.speed_test(self.student, dummy_input, pre_fix='stu_')
        self.speed_test(self.teacher, dummy_input, pre_fix='tea_')

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

    def forward(self, inputs):

        student_outs = self.student(inputs, self.need_return_para)
        teacher_outs = self.teacher(inputs, self.need_return_para)
        if self.norm:
            student_outs.last_representation /= student_outs.last_representation.norm(dim=-1, keepdim=True)
            teacher_outs.last_representation /= teacher_outs.last_representation.norm(dim=-1, keepdim=True)
        return student_outs, teacher_outs

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, self.hparams.model_type)
        self.log_info('train', loss, cal_res, batch_size=len(inputs))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, texts, sentence = batch
        if self.hparams.model_type == 'text':
            inputs, contrary_rep = texts, imgs
        else:
            inputs, contrary_rep = imgs, texts

        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, self.hparams.model_type)
        self.log_info('val', loss, cal_res, len(inputs))
        return {
            'student': self.all_gather(student_outs.last_representation),
            'teacher': self.all_gather(teacher_outs.last_representation),
            'contrary_rep': self.all_gather(contrary_rep)
        }

    def validation_step_end(self, step_out):
        return step_out

    def validation_epoch_end(self, outputs) -> None:
        stu_outs = []
        tea_outs = []
        contrary_reps = []
        for batch in outputs:
            student_out, teacher_out, contrary_rep = batch['student'], batch['teacher'], batch['contrary_rep']
            embedding = student_out.shape[-1]
            stu_outs.append(student_out.reshape(-1, embedding))
            tea_outs.append(teacher_out.reshape(-1, embedding))
            contrary_reps.append(contrary_rep.reshape(-1, embedding))
        stu_outs = torch.cat(stu_outs, dim=0).float()
        tea_outs = torch.cat(tea_outs, dim=0).float()
        contrary_reps = torch.cat(contrary_reps, dim=0).float()
        stu_logits, tea_logits = norm_and_logits(contrary_reps, stu_outs, tea_outs)[:2]

        self.log_acc(stu_logits, stage='val', prefix='stu')

        if self.current_epoch % 50 == 0:
            self.log_heatmap(stu_logits, stage='val', prefix='stu')

        if self.current_epoch == 0:
            self.log_heatmap(tea_logits, stage='val', prefix='tea')
            self.log_acc(tea_logits, stage='val', prefix='tea')

        return

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


def norm_and_logits(encode, stu_encode, tea_encode):
    encode = encode / encode.norm(dim=1, keepdim=True)
    encode = encode.float()
    stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
    tea_encode = tea_encode / tea_encode.norm(dim=1, keepdim=True)
    stu_logits = stu_encode @ encode.t()
    tea_logits = tea_encode @ encode.t()
    return stu_logits, tea_logits, stu_logits.T, tea_logits.T
