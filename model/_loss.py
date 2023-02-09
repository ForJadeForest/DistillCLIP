from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch import nn
from .loss_component import *
from .component.output import ControlOutput, CLIPOutput, VisionTransformerOutput, TextTransformerOutput

LOSSNAME = ['out_l1', 'out_ce', 'out_kl', 'out_cos', 'embedding_mse', 'attention_score_mse',
            'attention_probs_mse', 'hidden_rep_mse', 'attention_probs_kl', 'last_value_map_kl',
            'vit_kd', 'smd'
                      'hard_label', 'soft_label', 'fine_grain', 'logits_mse']

IMAGE_TEXT_LOSS = ['hard_label', 'soft_label', 'logits_mse', 'fine_grain']


class LossCalculator(nn.Module):
    def __init__(self, loss_name: List, loss_scale: dict = None,
                 temperature=None, percent=None, smd_tau: float = 0.04, vit_kd_para: Dict = None):
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

        self.temperature = temperature
        if vit_kd_para is not None:
            if 'low_layers_num' not in vit_kd_para:
                vit_kd_para['low_layers_num'] = 2
            if 'high_layers_num' not in vit_kd_para:
                vit_kd_para['high_layers_num'] = 1
        self.vit_kd_para = vit_kd_para
        self.smd_tau = smd_tau
        self.loss = self._init_loss()

        print(self.percent)
        print(self.loss_scale)

    def _init_loss(self):
        losses = nn.ModuleDict()

        for n in self.loss_name:
            if n == 'out_l1':
                loss_function = OutL1Loss()
            elif n == 'out_ce':
                loss_function = OutCELoss()
            elif n == 'out_kl':
                loss_function = OutKLLoss(self.temperature)
            elif n == 'out_cos':
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
                loss_function = SoftLabel(self.temperature)
            elif n == 'vit_kd':
                loss_function = ViTKDLoss(**self.vit_kd_para)
            elif n == 'logits_mse':
                loss_function = LogitsMSE()
            elif n == 'fine_grain':
                loss_function = FineGrainLoss()
            elif n == 'smd':
                loss_function = SMD(self.smd_tau)
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

        return need_para

    def cal_tow_tower_loss(self, stu_out: CLIPOutput, tea_out: CLIPOutput):
        cal_res = {}
        image_loss, image_loss_dict = self.cal_one_tower_loss(stu_out.visual_output, tea_out.visual_output)
        text_loss, text_loss_dict = self.cal_one_tower_loss(stu_out.text_output, tea_out.text_output)

        for k, v in image_loss_dict.items():
            cal_res['image_' + k] = v
        for k, v in text_loss_dict.items():
            cal_res['text_' + k] = v

        for loss_name in self.loss_name:
            loss = self.loss[loss_name]
            if loss_name == 'hard_label':
                cal_res[loss_name] = 0.5 * (loss(stu_out.i2t_logits) + loss(stu_out.t2i_logits))
            elif loss_name == 'soft_label':
                assert self.temperature
                logits_kl_loss = \
                    0.5 * (loss(stu_out.i2t_logits, tea_out.i2t_logits)
                           + loss(stu_out.t2i_logits, tea_out.t2i_logits)) * self.temperature ** 2
                cal_res[loss_name] = logits_kl_loss
            elif loss_name == 'logits_mse':
                cal_res[loss_name] = \
                    0.5 * (loss(stu_out.i2t_logits, tea_out.i2t_logits) + loss(stu_out.t2i_logits, tea_out.t2i_logits))
            elif loss_name == 'fine_grain':
                cal_res[loss_name] = loss(stu_out.visual_output.last_layer_output,
                                          stu_out.text_output.last_layer_output)

        loss = 0.5 * (image_loss + text_loss)
        for (loss_name, scale) in self.loss_scale.items():
            if loss_name in IMAGE_TEXT_LOSS:
                cal_res[loss_name] = cal_res[loss_name] * scale
                loss += cal_res[loss_name] * self.percent[loss_name]
        return loss, cal_res

    def cal_one_tower_loss(self,
                           stu_out: Union[VisionTransformerOutput, TextTransformerOutput],
                           tea_out: Union[VisionTransformerOutput, TextTransformerOutput]):
        cal_res = {}
        for loss_name in self.loss:
            loss = self.loss[loss_name]
            if loss_name == 'out_l1':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'out_ce':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'out_kl':
                assert self.temperature, 'You should give the temperature for the kl loss'
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
                assert self.vit_kd_para['low_layers_num'] + self.vit_kd_para['high_layers_num'] <= len(
                    stu_out.representations)
                stu_low_rep = torch.stack(stu_out.representations[:self.vit_kd_para['low_layers_num']], dim=1)
                tea_low_rep = torch.stack(tea_out.representations[:self.vit_kd_para['low_layers_num']], dim=1)
                stu_high_rep = torch.stack(stu_out.representations[-self.vit_kd_para['high_layers_num']:], dim=1)
                tea_high_rep = torch.stack(tea_out.representations[-self.vit_kd_para['high_layers_num']:], dim=1)

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
        if model_type == 'all':
            return self.cal_tow_tower_loss(stu_out, tea_out)
        else:
            return self.cal_one_tower_loss(stu_out, tea_out)

    def set_percent(self, new_percent):
        self.percent = new_percent

    def set_scale(self, new_scale):
        self.loss_scale = new_scale


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





















