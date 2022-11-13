from typing import *

import pytorch_lightning as pl
import torch
import transformers
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch import optim
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

from ._loss import LossCalculator
from ._utils import teacher_load
from .component.weight_share_model import RepeatVisionTransformer


class DistillModel(pl.LightningModule):
    def __init__(self, student_encoder: torch.nn.Module,
                 teacher_name: str, loss_control_para: Dict, download_root: str, norm: bool,
                 teacher_need_layers: List, model_type: str = 'text', map_type: str = 'mid', init_type=None,
                 warm_steps=10, total_steps=200, weight_decay=1e-3, lr: float = 1e-3):
        super().__init__()
        self.__dict__.update(locals())
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
        self.acc_metrics = []
        for k in self.k_list:
            self.acc_metrics.append(Accuracy(top_k=k))

    def on_train_start(self):
        if self.global_rank == 0:
            # 多gpu会报错
            if isinstance(self.logger, WandbLogger):
                self.logger.log_hyperparams({'student_para': self.student.hyper_para()})
                wandb.save('./*.py')
                wandb.save('./data/*.py')
                wandb.save('./model/*.py')
                wandb.save('./model/component/*.py')
                self.logger.watch(self)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.log_hyperparams(self.hparams, {"hp/stu_acc_top1": 0, "hp/stu_acc_top10": 0})

    def forward(self, inputs):
        if isinstance(self.student, RepeatVisionTransformer):
            student_outs = self.student(inputs, self.need_return_para)
        else:
            student_outs = self.student(inputs, self.need_return_para, only_last_state=False)
        teacher_outs = self.teacher(inputs, self.need_return_para, only_last_state=False)
        if self.norm:
            student_outs.last_representation /= student_outs.last_representation.norm(dim=-1, keepdim=True)
            teacher_outs.last_representation /= teacher_outs.last_representation.norm(dim=-1, keepdim=True)
        return student_outs, teacher_outs

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)

        loss, cal_res = self.loss_control(student_outs, teacher_outs, self.hparams.model_type)
        # Logging to TensorBoard by default
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
            'student': student_outs.last_representation,
            'teacher': teacher_outs.last_representation
        }


        # stu_logits, tea_logits = norm_and_logits(contrary_rep, student_outs.last_representation,
        #                                          teacher_outs.last_representation)[:2]

        # softmax_mean_score = torch.diagonal(torch.nn.functional.softmax(stu_logits, dim=1)).mean()
        # mean_score = torch.diagonal(stu_logits).mean()
        # self.log('softmax_mean_score', softmax_mean_score, batch_size=len(inputs), sync_dist=True)
        # self.log('mean_score', mean_score, batch_size=len(inputs), sync_dist=True)
        # # log metric
        # self.log('hp_metric', self.acc_metrics[0], metric_attribute='acc_metrics', batch_size=len(inputs))
        # for i, metric in enumerate(self.acc_metrics):
        #     metric.to(self.device)
        #     metric(stu_logits, label)
        #     self.log('hp_metric/stu_acc_top{}'.format(self.k_list[i]), metric, metric_attribute='acc_metrics',
        #              batch_size=len(inputs), sync_dist=True, )
        #     if self.current_epoch == 0:
        #         acc_tea = accuracy(tea_logits, label, top_k=self.k_list[i])
        #         self.log('hp_metric/tea_acc_top{}'.format(self.k_list[i]), acc_tea, prog_bar=False, sync_dist=True,
        #                  batch_size=len(inputs))
        #         tea_softmax_mean_score = torch.diagonal(torch.nn.functional.softmax(tea_logits, dim=1)).mean()
        #         tea_mean_score = torch.diagonal(tea_logits).mean()
        #         self.log('tea_softmax_mean_score', tea_softmax_mean_score, batch_size=len(inputs), sync_dist=True)
        #         self.log('tea_mean_score', tea_mean_score, batch_size=len(inputs), sync_dist=True)
        # Logging to TensorBoard by default

        # return loss

    def validation_step_end(self, step_out):
        student_out = step_out['student']
        teacher_out = step_out['teacher']

        #
        # # predictions from each GPU
        # predictions = batch_parts["pred"]
        # # losses from each GPU
        # losses = batch_parts["loss"]
        #
        # gpu_0_prediction = predictions[0]
        # gpu_1_prediction = predictions[1]
        #
        # # do something with both outputs
        # return (losses[0] + losses[1]) / 2
        return {
            'student': student_out,
            'teacher': teacher_out
        }

    def validation_epoch_end(self, outputs) -> None:
        a = 1
        # student_outs = torch.stack(outputs['student'])

    def log_info(self, stage, loss, cal_res, batch_size):

        self.log("{}/loss".format(stage), loss, batch_size=batch_size)
        for loss_name, loss_res in cal_res.items():
            self.log("{}/{}".format(stage, loss_name), loss_res, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warm_steps,
            num_training_steps=self.hparams.total_steps
        )
        """
        optimizer:
          class_path: AdamW
          init_args:
            lr: 1.0e-3
            weight_decay: 0.0001
            eps: 1.0e-8
        
        
        lr_scheduler:
          class_path: get_cosine_schedule_with_warmup
          init_args:
            num_warmup_steps: 5
            num_training_steps: 200
        """
        return [optimizer], [scheduler]


def norm_and_logits(encode, stu_encode, tea_encode):
    encode = encode / encode.norm(dim=1, keepdim=True)
    encode = encode.float()
    stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
    tea_encode = tea_encode / tea_encode.norm(dim=1, keepdim=True)
    stu_logits = stu_encode @ encode.t()
    tea_logits = tea_encode @ encode.t()
    return stu_logits, tea_logits, stu_logits.T, tea_logits.T
