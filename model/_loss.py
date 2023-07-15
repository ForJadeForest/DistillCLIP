from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import torch
import numpy as np
from torch import nn
from torch.nn import functional as f
from .loss_component import *
from .component.output import ControlOutput, CLIPOutput, VisionTransformerOutput, TextTransformerOutput
from .loss_component.utils import get_logits

LOSSNAME = ['out_l1', 'out_ce', 'out_kl', 'out_cos', 'embedding_mse', 'attention_score_mse',
            'attention_probs_mse', 'hidden_rep_mse', 'attention_probs_kl', 'last_value_map_kl',
            'vit_kd', 'smd'
                      'hard_label', 'soft_label', 'fine_grain', 'logits_mse']

IMAGE_TEXT_LOSS = ['hard_label', 'soft_label', 'logits_mse', 'fine_grain', 'SR']

NEED_SCALE = ['hard_label', 'soft_label']


class LossCalculator(nn.Module):
    def __init__(self, loss_name: List, temperature=0.07, t_learnable=True, clamp=4.6051,
                 loss_scale: dict = None, loss_init_args: Optional[Dict] = None,
                 percent=None, need_norm=False):
        super().__init__()
        self.loss_name = loss_name
        self.loss_scale = {}

        if loss_scale is None:
            loss_scale = {n: 1 for n in self.loss_name}
        for n in loss_name:
            self.loss_scale[n] = loss_scale.get(n, 1)

        if percent is None:
            percent = {n: 1 / len(loss_name) for n in self.loss_name}
        self.percent = percent
        default_value = (1 - sum(self.percent.values())) / len(self.percent)
        if len(loss_name) != len(self.percent.keys()) and default_value <= 0:
            raise ValueError(
                f"there are some loss default percent is negative. "
                f"Please check the sum of the percent {percent}"
                f"the default_value is {default_value} = (1 - sum(percent.values())) / len(percent)"
            )
        for n in loss_name:
            if n not in self.percent:
                self.percent[n] = default_value
        assert abs(sum(self.percent.values()) - 1) <= 1e-5

        if loss_init_args is None:
            self.loss_init_args = {}
        else:
            self.loss_init_args = loss_init_args
        if 'vit_kd' in self.loss_init_args:
            if 'low_layers_num' not in self.loss_init_args['vit_kd']:
                self.loss_init_args['vit_kd']['low_layers_num'] = 2
            if 'high_layers_num' not in self.loss_init_args['vit_kd']:
                self.loss_init_args['vit_kd']['high_layers_num'] = 1

        self.loss = self._init_loss()
        self._logit_scale = None
        self._need_norm = need_norm
        if set(NEED_SCALE) & set(loss_name):
            self._logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature), requires_grad=t_learnable)
            self._clamp = clamp
        print(self.percent)
        print(self.loss_scale)

    def get_logit_scale(self):
        if self._logit_scale is None:
            return {'logits_scale': 0.}

        logit_scale = self._logit_scale.detach()

        return {
            "logit_scale": logit_scale.exp(),
            "logit_scale_reciprocal": 1 / logit_scale.exp(),
        }

    def _init_loss(self):
        losses = nn.ModuleDict()

        for n in self.loss_name:
            init_args = self.loss_init_args.get(n)
            if n == 'out_l1':
                if init_args:
                    loss_function = OutL1Loss(**init_args)
                else:
                    loss_function = OutL1Loss()
            elif n == 'out_ce':
                loss_function = OutCELoss()
            elif n == 'out_kl':
                if init_args:
                    loss_function = OutKLLoss(**init_args)
                else:
                    loss_function = OutKLLoss()
            elif n == 'out_cos':
                if init_args:
                    loss_function = OutCosLoss(**init_args)
                else:
                    loss_function = OutCosLoss()
            elif n == 'embedding_mse':
                loss_function = EmbedMSELoss()
            elif n == 'attention_score_mse':
                loss_function = AttentionScoreMSE()
            elif n == 'attention_probs_mse':
                loss_function = AttentionProbsMSE()
            elif n == 'hidden_rep_mse':
                loss_function = HiddenMSE()
            elif n == 'attention_probs_kl':
                loss_function = AttentionProbsKL()
            elif n == 'last_value_map_kl':
                loss_function = LastValueMapKL()
            elif n == 'hard_label':
                loss_function = HardLabel()
            elif n == 'soft_label':
                if init_args:
                    loss_function = SoftLabel(**init_args)
                else:
                    loss_function = SoftLabel()
            elif n == 'vit_kd':
                loss_function = ViTKDLoss(**init_args)
            elif n == 'logits_mse':
                loss_function = LogitsMSE()
            elif n == 'fine_grain':
                loss_function = FineGrainLoss()
            elif n == 'smd':
                if init_args:
                    loss_function = SMD(**init_args)
                else:
                    loss_function = SMD()
            elif n == 'SR':
                loss_function = CLIPSR()
            else:
                raise ValueError("Invalid Loss Type!")
            losses[n] = loss_function
        return losses

    def get_control_output(self):
        need_para = ControlOutput()
        for n in self.loss_name:
            if n == 'embedding_mse':
                need_para.need_emb = True
            elif n == 'attention_score_mse':
                need_para.need_attn_score = True
            elif n == 'attention_probs_mse':
                need_para.need_attn_prob = True
            elif n == 'hidden_rep_mse':
                need_para.need_rep = True
            elif n == 'attention_probs_kl':
                need_para.attention_probs_mse = True
            elif n == 'last_value_map_kl':
                need_para.need_value_map = True
            elif n == 'vit_kd':
                need_para.need_rep = True

        return need_para

    def cal_tow_tower_loss(self, stu_out: CLIPOutput, tea_out: CLIPOutput):
        cal_res = {}
        image_loss, image_loss_dict = self.cal_one_tower_loss(stu_out.visual_output, tea_out.visual_output)
        text_loss, text_loss_dict = self.cal_one_tower_loss(stu_out.text_output, tea_out.text_output)

        for k, v in image_loss_dict.items():
            cal_res['image_' + k] = v
        for k, v in text_loss_dict.items():
            cal_res['text_' + k] = v

        if set(self.loss_name) & set(IMAGE_TEXT_LOSS):
            stu_logits_per_image, stu_logits_per_text = get_logits(stu_out, self._logit_scale)
            with torch.no_grad():
                tea_logits_per_image, tea_logits_per_text = get_logits(tea_out, self._logit_scale)
            for loss_name in self.loss_name:
                loss = self.loss[loss_name]
                if loss_name == 'hard_label':
                    cal_res[loss_name] = loss(stu_logits_per_image, stu_logits_per_text)
                elif loss_name == 'soft_label':
                    cal_res[loss_name] = loss(stu_logits_per_image, stu_logits_per_text,
                                              tea_logits_per_image, tea_logits_per_text)
                elif loss_name == 'logits_mse':
                    cal_res[loss_name] = loss(stu_logits_per_image, tea_logits_per_image)
                elif loss_name == 'fine_grain':
                    cal_res[loss_name] = loss(stu_out.visual_output.last_layer_output,
                                              stu_out.text_output.last_layer_output)
                elif loss_name == 'SR':
                    cal_res[loss_name] = loss(stu_logits_per_image, tea_logits_per_image)

        loss = 0.5 * (image_loss + text_loss)
        for (loss_name, scale) in self.loss_scale.items():
            if loss_name in IMAGE_TEXT_LOSS:
                cal_res[loss_name] = cal_res[loss_name] * scale
                loss += cal_res[loss_name] * self.percent[loss_name]
        return loss, cal_res

    def cal_one_tower_loss(self,
                           stu_out: Union[VisionTransformerOutput, TextTransformerOutput],
                           tea_out: Union[VisionTransformerOutput, TextTransformerOutput]):
        if self._need_norm:
            stu_out.last_representation = f.normalize(stu_out.last_representation, dim=-1)
            tea_out.last_representation = f.normalize(tea_out.last_representation, dim=-1)
        cal_res = {}
        for loss_name in self.loss:
            loss = self.loss[loss_name]
            if loss_name == 'out_l1':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'out_ce':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'out_kl':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'embedding_mse':
                cal_res[loss_name] = loss(stu_out.embedding, tea_out.embedding)
            elif loss_name == 'out_cos':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'attention_score_mse':
                cal_res[loss_name] = loss(stu_out.attention_scores, tea_out.attention_scores)
            elif loss_name == 'attention_probs_mse':
                cal_res[loss_name] = loss(stu_out.attention_probs, tea_out.attention_probs)
            elif loss_name == 'hidden_rep_mse':
                cal_res[loss_name] = loss(stu_out.representations, tea_out.representations)
            elif loss_name == 'attention_probs_kl':
                cal_res[loss_name] = loss(stu_out.attention_probs, tea_out.attention_probs)
            elif loss_name == 'last_value_map_kl':
                cal_res[loss_name] = loss(stu_out.value_map, tea_out.value_map)
            elif loss_name == 'vit_kd':
                assert self.loss_init_args['vit_kd']['low_layers_num'] + \
                       self.loss_init_args['vit_kd']['high_layers_num'] <= len(stu_out.representations)
                stu_low_rep = torch.stack(stu_out.representations[:self.loss_init_args['vit_kd']['low_layers_num']], dim=1)
                tea_low_rep = torch.stack(tea_out.representations[:self.loss_init_args['vit_kd']['low_layers_num']], dim=1)
                stu_high_rep = torch.stack(stu_out.representations[-self.loss_init_args['vit_kd']['high_layers_num']:], dim=1)
                tea_high_rep = torch.stack(tea_out.representations[-self.loss_init_args['vit_kd']['high_layers_num']:], dim=1)

                pred_s = [stu_low_rep, stu_high_rep]
                pred_t = [tea_low_rep, tea_high_rep]
                cal_res[loss_name] = loss(pred_s, pred_t)
            elif loss_name == 'smd':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
        loss = 0
        for (loss_name, scale) in self.loss_scale.items():
            if loss_name in IMAGE_TEXT_LOSS:
                continue
            cal_res[loss_name] = cal_res[loss_name] * scale
            loss += cal_res[loss_name] * self.percent[loss_name]

        return loss, cal_res

    def forward(self, stu_out: Union[CLIPOutput, VisionTransformerOutput, TextTransformerOutput],
                tea_out: Union[CLIPOutput, VisionTransformerOutput, TextTransformerOutput],
                model_type: str):
        if self._logit_scale:
            self._logit_scale.data = torch.clamp(self._logit_scale.data, 0, self._clamp)
        if model_type == 'all':
            return self.cal_tow_tower_loss(stu_out, tea_out)
        else:
            return self.cal_one_tower_loss(stu_out, tea_out)

    def set_percent(self, new_percent):
        self.percent = new_percent

    def set_scale(self, new_scale):
        self.loss_scale = new_scale


class MultiTeacherLossCalculator(nn.Module):
    """
    对每个teacher单独一个LossCalculator计算方法
    同时每个teacher的权重在这里面设置
    """

    def __init__(self, weight_method, loss_calculator_args):
        super().__init__()
        self.multi_loss_calculator = nn.ModuleDict()
        for t_n, args in loss_calculator_args.items():
            self.multi_loss_calculator[t_n] = LossCalculator(**args)
        self.teacher_name_list = list(loss_calculator_args.keys())
        self.weight_method = weight_method

    def forward(self, stu_out, tea_outs):
        total_loss = {}
        total_res = {}
        for t_n in self.teacher_name_list:
            tea_out = tea_outs[t_n]
            single_tea_loss, single_res = self.multi_loss_calculator[t_n](stu_out, tea_out, 'all')
            total_loss[t_n] = single_tea_loss
            total_res[t_n] = single_res

        loss = 0.
        if self.weight_method == 'sum':
            loss = sum(total_loss.values())
        elif self.weight_method == 'mean':
            loss = sum(total_loss.values()) / len(total_res)
        elif self.weight_method == 'ce_weight':
            weights = {}
            for t_n in self.teacher_name_list:
                assert 'hard_label' in total_res[t_n], f"if use ce_weight reduce method should use hard_label loss!"
                weights[t_n] = 1 / (1 + total_res[t_n]['hard_label'])
            for t_n in total_loss:
                loss += total_loss[t_n] * weights[t_n]
        else:
            raise ValueError(f"the weight_method value should be in ['sum', 'mean', 'ce_weight']")
        return loss, total_loss, total_res

    def get_control_output(self):
        output_control = ControlOutput()
        attr_name_list = ['need_emb', 'need_attn_score', 'need_value_map', 'need_attn_prob', 'need_rep']
        for t_n in self.teacher_name_list:
            single_control = self.multi_loss_calculator[t_n].get_control_output()
            for attr in attr_name_list:
                v = getattr(single_control, attr)
                if v:
                    setattr(output_control, attr, v)
        return output_control


@dataclass
class TotalLoss:
    out_l1 = nn.L1Loss()
    out_ce = nn.CrossEntropyLoss(reduction='mean')
    out_kl = nn.KLDivLoss(reduction='sum')
    out_cos = nn.CosineEmbeddingLoss()
    embedding_mse = nn.MSELoss()
    attention_score_mse = nn.MSELoss()
    attention_probs_mse = nn.MSELoss()
    hidden_rep_mse = nn.MSELoss()
    attention_probs_kl = nn.KLDivLoss(reduction='sum')
    last_value_map_kl = nn.KLDivLoss(reduction='sum')
    hard_label = nn.CrossEntropyLoss(reduction='mean')
    fine_grain = nn.CrossEntropyLoss(reduction='mean')
    soft_label = nn.KLDivLoss(reduction='sum')
    logits_mse = nn.MSELoss()
