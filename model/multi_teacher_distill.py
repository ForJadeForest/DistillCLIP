import copy
from typing import *

import pytorch_lightning as pl
import torch
import transformers
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import optim, nn
from torchmetrics.functional import accuracy
import torch.distributed as dist
from ._loss import LossCalculator
from .component import CLIPModel, ImageEncoder, CLIPOutput, MultiModalOutput, RepeatVisionTransformer, BascValMetric
from .utils import load_multi_teacher


def load_weight(image_student, text_student, load_path):
    def load_one_model(model: nn.Module, cpk: Optional[str]):
        if cpk is None:
            raise ValueError('the cpk is None! if you set the load_path parameter in model,'
                             ' you should give the image and text checkpoint path')
        save_res = torch.load(cpk)
        state_dict = {
            k.replace('student.', ''): v
            for k, v in save_res['state_dict'].items() if k.startswith('student')
        }

        model.load_state_dict(state_dict)
        return model

    image_student = load_one_model(image_student, load_path['image'])
    text_student = load_one_model(text_student, load_path['text'])
    return image_student, text_student


class MultiTeacherDistillModule(pl.LightningModule):
    def __init__(self, image_student: nn.Module, text_student: nn.Module, teacher_load_args_list: Dict[str, Dict],
                 loss_control_para: Dict, warm_steps, total_steps, weight_decay, lr: float,
                 validation_method: Dict[str, BascValMetric],
                 norm=False, freeze_embed: bool = False,
                 unfreeze_epoch: int = None, load_path: Dict = None, freeze_prefix: List = None):
        """

        :param image_student: Student image encoder
        :param text_student: Student Text encoder
        :param loss_control_para: To control which loss you want to use
        :param warm_steps: The Cos_lr_scheduler warm steps. It's the number of epoch
        :param total_steps: The total_epoch of training
        :param weight_decay: the weight_decay for lr
        :param lr: the learning rate
        :param download_root: The download path of CLIP Teacher model file
        :param norm: use the final output with l2 norm to calculate the loss
        :param teacher_name: The CLIP Teacher model name
        :param freeze_embed:  Whether to load the teacher embedding parameters and freeze them in training
        :param unfreeze_epoch: if is None, the freezed embedding will never unfreeze,
                               else, after the unfreeze_epoch, the embedding will unfreeze.
                               Only the freeze_embed is True will take effect
        :param load_path: the path for image encoder and text encoder checkpoint
        :param teacher_need_layers: the teacher layers you want to distillate
        :param freeze_prefix: A list, if the weight name startswith the elements in the list, it's weight will be freeze
        """
        super().__init__()
        self.save_hyperparameters(ignore=['image_student', 'text_student'])

        # define model
        if load_path:
            image_student, text_student = load_weight(image_student, text_student, load_path)
        self.student = CLIPModel(True, image_student, text_student, norm)

        self.teacher_dict = load_multi_teacher(self.device, teacher_load_args_list)
        self.teacher_name_list = self.teacher_dict.keys()
        for p in self.teacher_dict.parameters():
            p.requires_grad = False

        assert 'hard_label' in loss_control_para['loss_name'] or 'soft_label' in loss_control_para[
            'loss_name'], f"Now multi teacher distillation only support hard_label and soft_label loss"

        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output()
        if freeze_embed:
            self.freeze_image_embedding()
        self.unfreeze_epoch = unfreeze_epoch

        self.freeze_with_prefix(prefix_list=freeze_prefix)

        self.stu_val_method_metric = copy.deepcopy(validation_method)
        self.stu_val_method_metric = {
            i: v for i, (k, v) in enumerate(self.stu_val_method_metric.items())
        }

        self.multi_tea_val_method_metric = {}
        for teacher_name in self.teacher_name_list:
            self.multi_tea_val_method_metric[teacher_name] = copy.deepcopy(validation_method)
            for k, v in self.multi_tea_val_method_metric[teacher_name].items():
                v.set_model_name(teacher_name)
            self.multi_tea_val_method_metric[teacher_name] = {
                i: v for i, (k, v) in enumerate(self.multi_tea_val_method_metric[teacher_name].items())
            }
        for k, v in self.stu_val_method_metric.items():
            v.set_model_name('student')

    def on_train_start(self):
        self.logger_begin()

    @rank_zero_only
    def logger_begin(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_hyperparams({'student_para': self.student.hyper_para()})
            self.logger.experiment.log_code()
            wandb.define_metric(name='val/student-i2t-acc_top-1', summary='max')
            wandb.define_metric(name='val/student-i2t-acc_top-10', summary='max')
            wandb.define_metric(name='val/student-i2t-acc_top-50', summary='max')
            wandb.define_metric(name='val/student-t2i-acc_top-1', summary='max')
            wandb.define_metric(name='val/student-t2i-acc_top-10', summary='max')
            wandb.define_metric(name='val/student-t2i-acc_top-50', summary='max')

            wandb.define_metric(name='val/student-clip_score-human_rating-tau_c', summary='max')
            wandb.define_metric(name='val/student-ref_clip_score-human_rating-tau_c', summary='max')

    def forward(self, inputs) -> Tuple[CLIPOutput, Dict[str, MultiModalOutput]]:
        image, text = inputs
        student_outs: CLIPOutput = self.student(text, image, self.need_return_para)

        teacher_outs = {}
        for teacher in self.teacher_dict:
            teacher_outs[teacher] = self.teacher_dict[teacher](text, image)

        return student_outs, teacher_outs

    def on_train_epoch_start(self) -> None:
        if self.hparams.unfreeze_epoch:
            if self.current_epoch >= self.hparams.unfreeze_epoch:
                self.unfreeze_embed()
                self.hparams.unfreeze_epoch = None

    def log_data(self, data_info):
        self.log(f'{data_info["section"]}/{data_info["prefix"]}', data_info['value'], sync_dist=True,
                 add_dataloader_idx=False)

    def update_logits_in_dist(self, student_outs, teacher_outs):
        text_gather_output = self.all_gather(student_outs.text_output.last_representation, sync_grads=True)
        visual_gather_output = self.all_gather(student_outs.visual_output.last_representation, sync_grads=True)
        text_gather_output = text_gather_output.reshape(-1, text_gather_output.shape[-1])
        visual_gather_output = visual_gather_output.reshape(-1, visual_gather_output.shape[-1])

        student_outs.text_output.last_representation = text_gather_output
        student_outs.visual_output.last_representation = visual_gather_output

        student_outs.i2t_logits = visual_gather_output @ text_gather_output.t()
        student_outs.t2i_logits = student_outs.i2t_logits.t()

        for teacher in self.teacher_dict:
            text_gather_output = self.all_gather(teacher_outs[teacher].text_output.last_representation,
                                                 sync_grads=True)
            visual_gather_output = self.all_gather(teacher_outs[teacher].visual_output.last_representation,
                                                   sync_grads=True)
            text_gather_output = text_gather_output.reshape(-1, text_gather_output.shape[-1])
            visual_gather_output = visual_gather_output.reshape(-1, visual_gather_output.shape[-1])

            teacher_outs[teacher].text_output.last_representation = text_gather_output
            teacher_outs[teacher].visual_output.last_representation = visual_gather_output

            i2t_logits = text_gather_output @ visual_gather_output.t()
            t2i_logits = i2t_logits.t()

            teacher_outs[teacher].i2t_logits = i2t_logits
            teacher_outs[teacher].t2i_logits = t2i_logits

        return student_outs, teacher_outs

    def training_step(self, inputs, batch_idx):
        self.teacher_dict.eval()
        student_outs, teacher_outs = self.forward(inputs)
        if dist.is_initialized():
            student_outs, teacher_outs = self.update_logits_in_dist(student_outs, teacher_outs)
        loss = 0.
        for teacher_name in teacher_outs:
            single_teacher_loss, cal_res = self.loss_control(student_outs, teacher_outs[teacher_name], 'all')
            self.log_info(f'train_{teacher_name}_loss', single_teacher_loss, cal_res, batch_size=len(inputs))
            loss += single_teacher_loss

        loss /= len(teacher_outs)
        self.log('train_loss', loss, batch_size=len(inputs), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # batch: image, text
        method = self.stu_val_method_metric[dataloader_idx]
        step_res = method.validation_step(batch, self.student)
        for k, data_info in step_res.items():
            self.log_data(data_info)

        if self.current_epoch != 0:
            return
        for teacher_name in self.teacher_name_list:
            method = self.multi_tea_val_method_metric[teacher_name][dataloader_idx]
            step_res = method.validation_step(batch, self.teacher_dict[teacher_name])
            for k, data_info in step_res.items():
                self.log_data(data_info)

    def on_validation_epoch_end(self):
        # [gpu_num, batch, batch]
        for method_name, method in self.stu_val_method_metric.items():
            end_res = method.validation_end()
            for k, data_info in end_res.items():
                self.log_data(data_info)
            self.stu_val_method_metric[method_name].reset()

        if self.current_epoch != 0:
            return

        for teacher_name in self.teacher_name_list:
            for method_name, method in self.multi_tea_val_method_metric[teacher_name].items():
                end_res = method.validation_end()
                for k, data_info in end_res.items():
                    self.log_data(data_info)
                self.multi_tea_val_method_metric[teacher_name][method_name].reset()

    def log_info(self, section, loss, cal_res, batch_size):
        self.log(f"{section}/", loss, batch_size=batch_size, sync_dist=True)
        for loss_name, loss_res in cal_res.items():
            self.log(f"{section}/{loss_name}", loss_res, batch_size=batch_size, sync_dist=True)

    def configure_optimizers(self):
        opt_para = filter(lambda p: p.requires_grad, self.student.parameters())
        optimizer = optim.AdamW(opt_para, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warm_steps,
            num_training_steps=self.hparams.total_steps
        )
        return [optimizer], [scheduler]

    def unfreeze_embed(self):
        for n, p in self.student.named_parameters():
            p.requires_grad = True

    def freeze_with_prefix(self, prefix_list):
        if prefix_list is None:
            return

        for n, p in self.student.named_parameters():
            for prefix in prefix_list:
                if n.startswith(prefix):
                    print(f'freeze {n}')
                    p.requires_grad = False

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
