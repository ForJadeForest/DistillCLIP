from typing import *

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import cli as pl_cli
from torch import optim
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

try:
    from _utils import teacher_load, LayerMap, LossControl
    from clip_model import CLIPModel
except ModuleNotFoundError:
    from ._utils import teacher_load, LayerMap, LossControl


@pl_cli.MODEL_REGISTRY
class DualDistillModel(pl.LightningModule):
    def __init__(self, vit_paras: Dict, text_encoder_para: Dict, teacher_name: str, loss_control_para: Dict,
                 download_root: str, lr: float = 1e-3, map_type: str = 'mid', init_type=None, ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters(ignore=['student_encoder'])

        # 定义模型
        self.student = CLIPModel(True, vit_paras, text_encoder_para)
        self.teacher_name = teacher_name
        self.teacher, tea_layer_num = teacher_load(teacher_name, download_root, 'all')
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student.init_layers_with_teacher(self.layer_map, self.teacher.state_dict(), init_type)
        self.loss_control = LossControl(**loss_control_para)
        self.need_return_para = self.loss_control.need_output()
        # 定义指标
        self.k_list = [i for i in [1, 5, 10]]
        self.acc_metrics = []
        for k in self.k_list:
            self.acc_metrics.append(Accuracy(top_k=k))

    def on_train_start(self):
        if self.global_rank == 0:
            # 多gpu会报错
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.config.update({'student_para': self.student.hyper_para()})
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.log_hyperparams(self.hparams, {"hp/stu_acc_top1": 0, "hp/stu_acc_top10": 0})


    def forward(self, inputs):
        student_outs = self.student(inputs, only_last_state=False, **self.need_return_para)
        teacher_outs = self.teacher(inputs, only_last_state=False, **self.need_return_para)
        return student_outs, teacher_outs

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control.cal_tow_tower_loss(student_outs, teacher_outs, self.layer_map, self.device)
        # Logging to TensorBoard by default
        self.log_info('train', loss, cal_res, batch_size=len(inputs))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, texts, sentence = batch
        if self.hparams.model_type == 'text':
            inputs, contrary_rep = texts, imgs
        else:
            inputs, contrary = imgs, texts
            from clip import load
            clip_model, _ = load(self.teacher_name, device=self.device, download_root=self.hparams.download_root)
            tea_text_logits = clip_model.encode_text(contrary)
            contrary_rep = tea_text_logits

        student_outs, teacher_outs = self.forward(inputs)
        label = torch.arange(student_outs[0].shape[0], device=self.device)
        loss, cal_res = self.loss_control.cal_one_tower_loss(student_outs, teacher_outs, self.layer_map, self.device)
        stu_logits, tea_logits = norm_and_logits(contrary_rep, student_outs[0], teacher_outs[0])[:2]

        # log metric
        self.log('hp_metric', self.acc_metrics[0], metric_attribute='acc_metrics', batch_size=len(inputs))
        for i, metric in enumerate(self.acc_metrics):
            metric.to(self.device)
            metric(stu_logits, label)
            self.log('hp_metric/stu_acc_top{}'.format(self.k_list[i]), metric, metric_attribute='acc_metrics',
                     batch_size=len(inputs), sync_dist=True, )
            if self.current_epoch == 0:
                acc_tea = accuracy(tea_logits, label, top_k=self.k_list[i])
                self.log('hp_metric/tea_acc_top{}'.format(self.k_list[i]), acc_tea, prog_bar=False, sync_dist=True,
                         batch_size=len(inputs))
        # Logging to TensorBoard by default
        self.log_info('val', loss, cal_res, len(inputs))
        return loss

    def log_info(self, stage, loss, cal_res, batch_size):

        self.log("{}/loss".format(stage), loss, batch_size=batch_size)
        for loss_name, loss_res in cal_res.items():
            self.log("{}/{}".format(stage, loss_name), loss_res, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0001)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=180)
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
