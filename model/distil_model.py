from typing import *

import pytorch_lightning as pl
import torch
import transformers
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import optim, nn
from torchmetrics.functional import accuracy

from ._loss import LossCalculator
from ._metrics import cal_speed, cal_flop
from ._utils import teacher_load
from .component.image_encoder import ImageEncoder
from .component.weight_share_model import RepeatVisionTransformer


class DistillModel(pl.LightningModule):
    def __init__(self, student_encoder: torch.nn.Module,
                 teacher_name: str, loss_control_para: Dict, download_root: str, freeze_embed: bool = False,
                 teacher_need_layers: List = None, model_type: str = 'image', map_type: str = None, init_type=None,
                 warm_steps=10, total_steps=200, weight_decay=1e-3, lr: float = 1e-3, norm: bool = False,
                 unfreeze_epoch=None):
        super().__init__()
        if model_type not in ['text', 'image']:
            raise ValueError(f"the model_type should in ['text', 'image'], bug got {model_type}")
        self.save_hyperparameters(ignore=['student_encoder'])

        self.student = student_encoder
        self.teacher_name = teacher_name
        self.teacher = teacher_load(teacher_name, download_root, model_type, need_layers=teacher_need_layers)
        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output()

        if isinstance(self.student, ImageEncoder) and len(self.teacher.need_layers) != len(self.student.need_layers):
            raise ValueError(
                f'the teacher need_layers length is not equal to student need_layers length. '
                f'But get teacher: {self.teacher.need_layers}, student: {self.student.need_layers}')

        for p in self.teacher.parameters():
            p.requires_grad = False
        if model_type == 'image' and freeze_embed:
            self.freeze_image_embedding()

        # define metric
        self.k_list = [i for i in [1, 3, 5, 10, 20, 50]]

    def on_train_start(self):
        self.logger_begin()
        # if self.hparams.model_type == 'image':
        #     dummy_input = torch.rand(size=(1, 3, 224, 224), device=self.device)
        # else:
        #     dummy_input = torch.rand(size=(1, 77), device=self.device)
        # self.speed_test(self.student, dummy_input, prefix='stu_')
        # self.speed_test(self.teacher, dummy_input, prefix='tea_')

    @rank_zero_only
    def logger_begin(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_hyperparams({'student_para': self.student.hyper_para()})
            self.logger.experiment.log_code()
            wandb.define_metric(name='val_stu_acc/stu_acc_top1', summary='max')
            wandb.define_metric(name='val_stu_acc/stu_acc_top10', summary='max')
            wandb.define_metric(name='val_stu_acc/stu_acc_top50', summary='max')
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(self.hparams, {"hp/stu_acc_top1": 0, "hp/stu_acc_top10": 0})

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
        self.log_dict(metric_dict, sync_dist=True)

    def forward(self, inputs):

        student_outs = self.student(inputs, self.need_return_para)
        with torch.no_grad():
            teacher_outs = self.teacher(inputs, self.need_return_para)
        if self.hparams.norm:
            student_outs.last_representation /= student_outs.last_representation.norm(dim=-1, keepdim=True)
            teacher_outs.last_representation /= teacher_outs.last_representation.norm(dim=-1, keepdim=True)
        return student_outs, teacher_outs

    def on_train_epoch_start(self) -> None:
        if self.hparams.unfreeze_epoch:
            if self.current_epoch >= self.unfreeze_epoch:
                self.unfreeze_embed()
                self.hparams.unfreeze_epoch = False

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, self.hparams.model_type)
        self.log_info('train_loss', loss, cal_res, batch_size=len(inputs))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, texts, _ = batch
        if self.hparams.model_type == 'text':
            inputs, contrary_rep = texts, imgs
        else:
            inputs, contrary_rep = imgs, texts

        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, self.hparams.model_type)

        stu_logits, tea_logits = norm_and_logits(
            contrary_rep, student_outs.last_representation, teacher_outs.last_representation)[:2]

        self.log_info('val_loss', loss, cal_res, batch_size=len(batch))
        self.log_acc(stu_logits, section='val_step', prefix='stu')
        self.log_acc(tea_logits, section='val_step', prefix='tea')
        self.log_diag_score(stu_logits, section='val_step', prefix='stu')

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

        self.log_acc(stu_logits, section='val_stu_acc', prefix='stu')
        self.log_diag_score(stu_logits, section='val_stu_score', prefix='stu')

        if self.current_epoch == 0:
            self.log_diag_score(tea_logits, section='val_tea_score', prefix='tea')
            self.log_acc(tea_logits, section='val_tea_acc', prefix='tea')
        return

    def log_info(self, section, loss, cal_res, batch_size):

        self.log("{}/loss".format(section), loss, batch_size=batch_size, sync_dist=True)
        for loss_name, loss_res in cal_res.items():
            self.log("{}/{}".format(section, loss_name), loss_res, batch_size=batch_size, sync_dist=True)

    def configure_optimizers(self):
        opt_para = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(opt_para, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warm_steps,
            num_training_steps=self.hparams.total_steps
        )
        return [optimizer], [scheduler]

    def log_diag_score(self, logits, section, prefix):
        softmax_logits = nn.functional.softmax(logits, dim=1)
        softmax_mean_score = torch.diagonal(softmax_logits).mean()
        mean_score = torch.diagonal(logits).mean()

        self.log(f'{section}/{prefix}_softmax_mean_score', softmax_mean_score, batch_size=logits.shape[0],
                 sync_dist=True)
        self.log(f'{section}/{prefix}_mean_score', mean_score, batch_size=logits.shape[0], sync_dist=True)

    def log_heatmap(self, logits, section, prefix):
        softmax_logits = nn.functional.softmax(logits, dim=1)
        self.logger.log_image(key=f"{section}_{prefix}_logits",
                              images=[plt.imshow(logits.detach().cpu()),
                                      plt.imshow(softmax_logits.detach().cpu())],
                              caption=[f"{section}_{prefix}_map", f"{section}_{prefix}_softmax_map"])

    def log_acc(self, logits, section, prefix):
        label = torch.arange(logits.shape[0], device=self.device)
        for k in self.k_list:
            acc = accuracy(logits, label, top_k=k, task='multiclass', num_classes=logits.shape[0])
            self.log(f'{section}/{prefix}_acc_top{k}', acc, batch_size=logits.shape[0], sync_dist=True)

    def unfreeze_embed(self):
        for n, p in self.student.named_parameters():
            p.requires_grad = True

    def freeze_image_embedding(self):
        student_weights = self.student.state_dict()
        if isinstance(self.student, RepeatVisionTransformer):
            stu_keys = ['patch_embed.proj.weight', 'cls_token', 'pos_embed']
            tea_keys = ['visual.conv1.weight', 'visual.class_embedding', 'visual.positional_embedding']
            for s_k, t_k in zip(stu_keys, tea_keys):
                weights = self.teacher.state_dict()[t_k]
                if 'cls_token' in s_k:
                    weights = weights.unsqueeze(0).unsqueeze(0)
                if 'pos_embed' in s_k:
                    weights = weights.unsqueeze(0)
                student_weights[s_k] = weights

            self.student.load_state_dict(student_weights)
            for n, p in self.student.named_parameters():
                if n in stu_keys:
                    p.requires_grad = False
        elif isinstance(self.student, ImageEncoder):
            freeze_key = ['visual.conv1.weight', 'visual.class_embedding', 'visual.positional_embedding']
            for k in freeze_key:
                student_weights[k] = self.teacher.state_dict()[k]
            self.student.load_state_dict(student_weights)
            for n, p in self.student.named_parameters():
                if n in freeze_key:
                    p.requires_grad = False


def norm_and_logits(encode, stu_encode, tea_encode):
    encode = encode / encode.norm(dim=1, keepdim=True)
    encode = encode.float()
    stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
    tea_encode = tea_encode / tea_encode.norm(dim=1, keepdim=True)
    stu_logits = stu_encode @ encode.t()
    tea_logits = tea_encode @ encode.t()
    return stu_logits, tea_logits, stu_logits.T, tea_logits.T
