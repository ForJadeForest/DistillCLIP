import copy
from typing import *

import torch
import transformers
import wandb

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import optim, nn

from ._loss import LossCalculator
from .component import CLIPModel, ImageEncoder, CLIPOutput, RepeatVisionTransformer, BascValMetric
from .utils import teacher_load


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


class DualDistillModel(LightningModule):
    def __init__(self, image_student: nn.Module, text_student: nn.Module,
                 loss_control_para: Dict, warm_steps, total_steps, weight_decay, lr: float,
                 validation_method: Dict[str, BascValMetric],
                 download_root: str, norm=False, teacher_name: str = 'ViT-B/32', freeze_embed: bool = False,
                 unfreeze_epoch: int = None, load_path: Dict = None, teacher_need_layers: List = None,
                 freeze_prefix: List = None):
        """

        :param image_student: Student image encoder
        :param text_student: Student Text encoder
        :param loss_control_para: To control which loss you want to use
        :param warm_steps: The Cos_lr_scheduler warm steps. It's the number of epoch.
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

        self.teacher = teacher_load(teacher_name, download_root, 'all', need_layers=teacher_need_layers)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output()
        if freeze_embed:
            self.freeze_image_embedding()
        self.unfreeze_epoch = unfreeze_epoch

        self.freeze_with_prefix(prefix_list=freeze_prefix)
        # define acc top k
        self.k_list = [i for i in [1, 3, 5, 10, 20, 50]]

        val_method_name = validation_method.keys()
        self.val_method_map = {
            i: method for i, method in enumerate(val_method_name)
        }
        self.stu_val_method_metric = copy.deepcopy(validation_method)
        self.tea_val_method_metric = copy.deepcopy(validation_method)
        for k, v in self.stu_val_method_metric.items():
            v.set_model_name('student')
        for k, v in self.tea_val_method_metric.items():
            v.set_model_name('teacher')

        self.stu_val_method_metric = {
            i: v for i, (k, v) in enumerate(self.stu_val_method_metric.items())
        }
        self.tea_val_method_metric = {
            i: v for i, (k, v) in enumerate(self.tea_val_method_metric.items())
        }

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

    def forward(self, inputs) -> Tuple[CLIPOutput, CLIPOutput]:
        image, text = inputs
        student_outs: CLIPOutput = self.student(text, image, self.need_return_para)
        teacher_outs: CLIPOutput = self.teacher(text, image, self.need_return_para)
        if self.hparams.norm:
            student_outs, teacher_outs = norm_last_representation(student_outs, teacher_outs)
        return student_outs, teacher_outs

    def on_train_epoch_start(self) -> None:
        if self.hparams.unfreeze_epoch:
            if self.current_epoch >= self.hparams.unfreeze_epoch:
                self.unfreeze_embed()
                self.hparams.unfreeze_epoch = None

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)

        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        self.log_info('train_loss', loss, cal_res, batch_size=len(inputs))

        return loss

    def log_data(self, data_info):
        self.log(f'{data_info["section"]}/{data_info["prefix"]}', data_info['value'], sync_dist=True,
                 add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        method = self.stu_val_method_metric[dataloader_idx]
        step_res = method.validation_step(batch, self.student)
        for k, data_info in step_res.items():
            self.log_data(data_info)

        if self.current_epoch == 0:
            method = self.tea_val_method_metric[dataloader_idx]
            step_res = method.validation_step(batch, self.teacher)
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

        for method_name, method in self.tea_val_method_metric.items():
            end_res = method.validation_end()
            for k, data_info in end_res.items():
                self.log_data(data_info)
            self.tea_val_method_metric[method_name].reset()

    def log_info(self, section, loss, cal_res, batch_size):
        self.log(f"{section}/loss", loss, batch_size=batch_size, sync_dist=True)
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


def norm_last_representation(stu_outs, tea_outs):
    stu_outs.visual_output.last_representation /= stu_outs.visual_output.last_representation.norm(dim=-1, keepdim=True)
    stu_outs.text_output.last_representation /= stu_outs.text_output.last_representation.norm(dim=-1, keepdim=True)
    tea_outs.visual_output.last_representation /= tea_outs.visual_output.last_representation.norm(dim=-1, keepdim=True)
    tea_outs.text_output.last_representation /= tea_outs.text_output.last_representation.norm(dim=-1, keepdim=True)

    return stu_outs, tea_outs
