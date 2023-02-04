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
from ._metrics import cal_flop, cal_speed
from .utils import teacher_load
from .component.clip_model import CLIPModel
from .component.image_encoder import ImageEncoder
from .component.output import CLIPOutput
from .component.weight_share_model import RepeatVisionTransformer


def load_weight(image_student, text_student, load_path):
    def load_one_model(model: nn.Module, cpk: Optional[str]):
        if cpk is None:
            return model
        save_res = torch.load(cpk)
        model.load_state_dict(save_res['state_dic'])
        return model

    image_student = load_one_model(image_student, load_path['image'])
    text_student = load_one_model(text_student, load_path['text'])
    return image_student, text_student


class DualDistillModel(pl.LightningModule):
    def __init__(self, image_student: nn.Module, text_student: nn.Module, teacher_need_layers: List, teacher_name: str,
                 loss_control_para: Dict, freeze_embed: bool, warm_steps, total_steps, weight_decay, lr: float,
                 download_root: str, norm=False, unfreeze_epoch: int = None, load_path: Dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=['image_student', 'text_student'])

        # define model
        if load_path:
            image_student, text_student = load_weight(image_student, text_student, load_path)
        self.student = CLIPModel(True, image_student, text_student, norm)
        self.teacher_name = teacher_name
        self.teacher = teacher_load(teacher_name, download_root, 'all', need_layers=teacher_need_layers)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output()
        if freeze_embed:
            self.freeze_image_embedding()

        # define metric
        self.k_list = [i for i in [1, 3, 5, 10, 20, 50]]

    def on_train_start(self):
        self.logger_begin()
        # dummy_input = (torch.randint(high=49407, size=(1, 77), device=self.device),
        #                torch.rand(size=(1, 3, 224, 224), device=self.device))
        # self.speed_test(self.student, dummy_input, prefix='stu')
        # self.speed_test(self.teacher, dummy_input, prefix='tea')

    @rank_zero_only
    def logger_begin(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_hyperparams({'student_para': self.student.hyper_para()})
            self.logger.experiment.log_code()
            wandb.define_metric(name='val_stu_acc/stu_acc_top1', summary='max')
            wandb.define_metric(name='val_stu_acc/stu_acc_top10', summary='max')
            wandb.define_metric(name='val_stu_acc/stu_acc_top50', summary='max')
            wandb.define_metric(name='val_stu_image_tea_text/stu_image_tea_text', summary='max')
            wandb.define_metric(name='val_stu_text_tea_image/stu_text_tea_image', summary='max')

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

    def forward(self, inputs) -> Tuple[CLIPOutput, CLIPOutput]:
        image, text = inputs
        student_outs: CLIPOutput = self.student(text, image, self.need_return_para)
        teacher_outs: CLIPOutput = self.teacher(text, image, self.need_return_para)
        if self.hparams.norm:
            student_outs, teacher_outs = norm_last_representation(student_outs, teacher_outs)
        return student_outs, teacher_outs

    def on_train_epoch_start(self) -> None:
        if self.hparams.unfreeze_epoch:
            if self.current_epoch >= self.unfreeze_epoch:
                self.unfreeze_embed()
                self.hparams.unfreeze_epoch = False

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)

        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        self.log_info('train_loss', loss, cal_res, batch_size=len(inputs))

        # stu_logits, _ = norm_and_logits(student_outs.visual_output.last_representation,
        #                                 student_outs.text_output.last_representation)
        # tea_logits, _ = norm_and_logits(teacher_outs.visual_output.last_representation,
        #                                 teacher_outs.text_output.last_representation)
        # self.log_acc(stu_logits, section='train_acc', prefix='stu')
        # self.log_acc(tea_logits, section='train_acc', prefix='tea')
        # self.log_diag_score(stu_logits, section='train_diag_score', prefix='stu')

        return loss

    def validation_step(self, batch, batch_idx):
        student_outs, teacher_outs = self.forward(batch)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        stu_logits, _ = norm_and_logits(student_outs.visual_output.last_representation,
                                        student_outs.text_output.last_representation)
        tea_logits, _ = norm_and_logits(teacher_outs.visual_output.last_representation,
                                        teacher_outs.text_output.last_representation)

        self.log_info('val_loss', loss, cal_res, batch_size=len(batch))
        self.log_acc(stu_logits, section='val_step', prefix='stu')
        self.log_acc(tea_logits, section='val_step', prefix='tea')
        self.log_diag_score(stu_logits, section='val_step', prefix='stu')

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

        self.log_acc(stu_logits, section='val_stu_acc', prefix='stu')
        self.log_acc(stu_image_tea_text_logits, section='val_stu_image_tea_text', prefix='stu_image_tea_text')
        self.log_acc(stu_text_tea_image_logits, section='val_stu_text_tea_image', prefix='stu_text_tea_image')
        self.log_diag_score(stu_logits, section='val_stu_score', prefix='stu')

        if self.current_epoch == 0:
            self.log_diag_score(tea_logits, section='val_tea_score', prefix='tea')
            self.log_acc(tea_logits, section='val_tea_acc', prefix='tea')

        return

    def log_info(self, section, loss, cal_res, batch_size):
        self.log(f"{section}/loss", loss, batch_size=batch_size, sync_dist=True)
        for loss_name, loss_res in cal_res.items():
            self.log(f"{section}/{loss_name}", loss_res, batch_size=batch_size, sync_dist=True)

    def configure_optimizers(self):
        opt_para = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(opt_para, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
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
            acc = accuracy(logits, label, top_k=k)
            self.log(f'{section}/{prefix}_acc_top{k}', acc, batch_size=logits.shape[0], sync_dist=True)

    def unfreeze_embed(self):
        for n, p in self.student.named_parameters():
            p.requires_grad = True

    def freeze_image_embedding(self):
        freeze_key = ['visual.conv1.weight', 'visual.class_embedding', 'visual.positional_embedding']
        teacher_keys = ['image_encoder.' + k for k in freeze_key]
        if isinstance(self.student.image_encoder, RepeatVisionTransformer):
            student_weights = self.student.state_dict()
            base_key = ['patch_embed.proj.weight', 'cls_token', 'pos_embed']
            student_keys = ['image_encoder.' + k for k in base_key]

            for s_k, t_k in zip(student_keys, teacher_keys):
                weights = self.teacher.state_dict()[t_k]
                if 'cls_token' in s_k:
                    weights = weights.unsqueeze(0).unsqueeze(0)
                if 'pos_embed' in s_k:
                    weights = weights.unsqueeze(0)
                student_weights[s_k] = weights

            self.student.load_state_dict(student_weights)
            for n, p in self.student.named_parameters():
                if n in student_keys:
                    p.requires_grad = False
        elif isinstance(self.student.image_encoder, ImageEncoder):
            student_keys = teacher_keys
            student_weights = self.student.state_dict()
            for k in teacher_keys:
                student_weights[k] = self.teacher.state_dict()[k]
            self.student.load_state_dict(student_weights)
            for n, p in self.student.named_parameters():
                if n in student_keys:
                    p.requires_grad = False


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
