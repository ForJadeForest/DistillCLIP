from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as f

from model._utils import ControlOutput

LOSSNAME = ['out_l1', 'out_ce', 'out_kl', 'out_cos', 'embedding_mse', 'attention_score_mse',
            'attention_probs_mse', 'hidden_rep_mse', 'attention_probs_kl', 'last_value_map_kl',
            'vit_kd',
            'hard_label', 'soft_label']


class LossCalculator(nn.Module):
    def __init__(self, loss_name, loss_scale: dict = None,
                 temperature=None, percent=None, vit_kd_para: Dict = None, normlize=True):
        super().__init__()
        self.loss_name = loss_name
        self.loss_scale = loss_scale
        if self.loss_scale is None:
            self.loss_scale = {n: 1 for n in self.loss_name}
        self.percent = percent
        if self.percent is None:
            self.percent = {n: 1 / len(loss_name) for n in loss_name}
        self.temperature = temperature
        if not 'low_layers_num' in vit_kd_para:
            vit_kd_para['low_layers_num'] = 2
        if not 'high_layers_num' in vit_kd_para:
            vit_kd_para['high_layers_num'] = 1
        self.vit_kd_para = vit_kd_para
        self.loss = self._init_loss()
        self.normlize = normlize
        assert abs(sum([v for v in self.percent.values()]) - 1) <= 1e-5
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
                loss_function = OutKLLoss(self.temperature, )
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
                loss_function = SoftLabel()
            elif n == 'vit_kd':
                loss_function = ViTKDLoss(**self.vit_kd_para)
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

    def cal_tow_tower_loss(self, stu_out, tea_out):
        stu_image_output, stu_text_output, stu_logits = stu_out
        tea_image_output, tea_text_output, tea_logits = tea_out
        cal_res = {}
        image_loss, image_loss_dict = self.cal_one_tower_loss(stu_image_output, tea_image_output)
        text_loss, text_loss_dict = self.cal_one_tower_loss(stu_text_output, tea_text_output)
        cal_res.update(image_loss_dict)
        for k, v in image_loss_dict.items():
            cal_res['image_' + k] = v
        for k, v in text_loss_dict.items():
            cal_res['text_' + k] = v

        for loss_name in self.loss_name:
            loss = self.loss[loss_name]
            if loss_name == 'label':
                label = torch.arange(stu_logits.shape[0], device=stu_logits.device)
                cal_res[loss_name] = loss(stu_logits, label)
            elif loss_name == 'soft_label':
                assert self.temperature
                logits_kl_loss = loss(
                    f.softmax(stu_logits / self.temperature, dim=1).log(),
                    f.softmax(tea_logits / self.temperature, dim=1)
                ) * self.temperature ** 2
                cal_res[loss_name] = logits_kl_loss
        loss = 0.5 * (image_loss + text_loss)
        for (loss_name, scale) in self.loss_scale.items():
            if loss_name == 'label' or loss_name == 'soft_label':
                cal_res[loss_name] = cal_res[loss_name] * scale
            loss += cal_res[loss_name] * self.percent[loss_name]
        return loss, cal_res

    def cal_one_tower_loss(self, stu_out, tea_out):
        cal_res = {}
        for loss_name in self.loss:
            loss = self.loss[loss_name]
            if loss_name == 'out_l1':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'out_ce':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'out_kl':
                # calculate the pred loss
                assert self.temperature, 'You should give the temperature for the kl loss'

                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'embedding_mse':
                # calculate the embedding loss
                cal_res[loss_name] = loss(stu_out.embedding, tea_out.embedding)
            elif loss_name == 'out_cos':
                cal_res[loss_name] = loss(stu_out.last_representation, tea_out.last_representation)
            elif loss_name == 'attention_score_mse':
                # attention loss
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
                stu_low_rep = torch.stack(stu_out.representations[:self.vit_kd_para['low_layers_num']], dim=1)
                tea_low_rep = torch.stack(tea_out.representations[:self.vit_kd_para['low_layers_num']], dim=1)
                stu_high_rep = torch.stack(stu_out.representations[-self.vit_kd_para['high_layers_num']:], dim=1)
                tea_high_rep = torch.stack(tea_out.representations[-self.vit_kd_para['high_layers_num']:], dim=1)

                pred_s = [stu_low_rep, stu_high_rep]
                pred_t = [tea_low_rep, tea_high_rep]
                cal_res[loss_name] = loss(pred_s, pred_t)
            if self.normlize:
                cal_res[loss_name] /= cal_res[loss_name]
        loss = 0
        for (loss_name, scale) in self.loss_scale.items():
            if loss_name == 'label' or loss_name == 'soft_label':
                continue
            cal_res[loss_name] = cal_res[loss_name] * scale
            loss += cal_res[loss_name] * self.percent[loss_name]

        return loss, cal_res

    def forward(self, stu_out, tea_out, model_type):
        if model_type == 'all':
            return self.cal_tow_tower_loss(stu_out, tea_out)
        else:
            return self.cal_one_tower_loss(stu_out, tea_out)


@dataclass
class TotalLoss:
    out_l1 = nn.L1Loss()
    out_ce = nn.CrossEntropyLoss(reduction='mean')
    out_kl = nn.KLDivLoss(reduction='batchmean')
    out_cos = nn.CosineEmbeddingLoss()
    embedding_mse = nn.MSELoss()
    attention_score_mse = nn.MSELoss()
    attention_probs_mse = nn.MSELoss()
    hidden_rep_mse = nn.MSELoss()
    attention_probs_kl = nn.KLDivLoss(reduction='batchmean')
    last_value_map_kl = nn.KLDivLoss(reduction='batchmean')
    hard_label = nn.CrossEntropyLoss(reduction='mean')
    soft_label = nn.KLDivLoss(reduction='batchmean')
    vit_kd = None


class OutL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = TotalLoss.out_l1

    def forward(self, stu_out, tea_out):
        return self.loss(stu_out, tea_out)


class OutCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = TotalLoss.out_ce

    def forward(self, stu_out, tea_out):
        return self.loss(
            stu_out,  # [batch, out_dim]
            tea_out.softmax(dim=1)
        )


class OutKLLoss(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.loss = TotalLoss.out_kl

        self.temperature = t

    def forward(self, stu_out, tea_out):
        return self.loss(
            f.log_softmax(stu_out / self.temperature, dim=1),
            f.softmax(tea_out / self.temperature, dim=1)
        ) * self.temperature ** 2


class OutCosLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = TotalLoss.out_cos

    def forward(self, stu_out: torch.Tensor, tea_out):
        return self.loss(stu_out, tea_out, torch.ones(len(stu_out), device=stu_out.device))


class EmbedMSELoss(nn.Module):
    def __init__(self):
        super(EmbedMSELoss, self).__init__()
        self.loss = TotalLoss.embedding_mse

    def forward(self, stu_embedding, tea_embedding):
        return self.loss(stu_embedding, tea_embedding)


class AttentionScoreMSE(nn.Module):
    def __init__(self):
        super(AttentionScoreMSE, self).__init__()
        self.loss = TotalLoss.attention_score_mse

    def forward(self, stu_attn_score, tea_attn_score):
        res_loss = 0
        for layer_num, (stu_out, tea_out) in enumerate(zip(stu_attn_score, tea_attn_score)):
            stu_head_num = stu_out.shape[1]
            tea_head_num = tea_out.shape[1]
            stu_mean_head_out = torch.sum(stu_out, dim=1) / stu_head_num
            tea_mean_head_out = torch.sum(tea_out, dim=1) / tea_head_num
            if layer_num == 0:
                res_loss = self.loss(stu_mean_head_out, tea_mean_head_out)
            else:
                res_loss += self.loss(stu_mean_head_out, tea_mean_head_out)
        res_loss /= len(stu_attn_score)
        return res_loss


class AttentionProbsMSE(nn.Module):
    def __init__(self):
        super(AttentionProbsMSE, self).__init__()
        self.loss = TotalLoss.attention_probs_mse

    def forward(self, stu_attn_probs, tea_attn_probs):
        res_loss = 0
        for layer_num, (stu_out, tea_out) in enumerate(zip(stu_attn_probs, tea_attn_probs)):
            stu_head_num = stu_out.shape[1]
            tea_head_num = tea_out.shape[1]
            stu_mean_head_out = torch.sum(stu_out, dim=1) / stu_head_num
            tea_mean_head_out = torch.sum(tea_out, dim=1) / tea_head_num
            if layer_num == 0:
                res_loss = self.loss(stu_mean_head_out, tea_mean_head_out)
            else:
                res_loss += self.loss(stu_mean_head_out, tea_mean_head_out)
        res_loss /= len(stu_attn_probs)
        return res_loss


class HiddenMSE(nn.Module):
    def __init__(self):
        super(HiddenMSE, self).__init__()
        self.loss = TotalLoss.hidden_rep_mse

    def forward(self, stu_hidden, tea_hidden):
        res_loss = 0
        for layer_num, (stu_out, tea_out) in enumerate(zip(stu_hidden, tea_hidden)):
            if layer_num == 0:
                res_loss = self.loss(stu_out, tea_out)
            else:
                res_loss += self.loss(stu_out, tea_out)
        res_loss /= len(stu_hidden)
        return res_loss


class AttentionProbsKL(nn.Module):
    def __init__(self):
        super(AttentionProbsKL, self).__init__()
        self.loss = TotalLoss.attention_probs_kl

    def forward(self, stu_attn_probs, tea_attn_probs):
        res_loss = 0
        for layer_num, (stu_out, tea_out) in enumerate(zip(stu_attn_probs, tea_attn_probs)):
            stu_head_num = stu_out.shape[1]
            tea_head_num = tea_out.shape[1]
            stu_mean_head_out = torch.sum(stu_out, dim=1) / stu_head_num
            tea_mean_head_out = torch.sum(tea_out, dim=1) / tea_head_num
            if layer_num == 0:
                res_loss = self.loss(stu_mean_head_out.log(), tea_mean_head_out)
            else:
                res_loss += self.loss(stu_mean_head_out.log(), tea_mean_head_out)
        res_loss /= len(stu_attn_probs)
        return res_loss


class LastValueMapKL(nn.Module):
    def __init__(self):
        super(LastValueMapKL, self).__init__()
        self.loss = TotalLoss.last_value_map_kl

    def forward(self, stu_value_map, tea_value_map):
        return self.loss(
            f.softmax(stu_value_map, dim=1).log(),
            f.softmax(tea_value_map, dim=1)
        )


class HardLabel(nn.Module):
    def __init__(self):
        super(HardLabel, self).__init__()
        self.loss = TotalLoss.hard_label

    def forward(self, stu_logits):
        label = torch.arange(stu_logits.shape[0], device=stu_logits.device)
        return self.loss(stu_logits, label)


class SoftLabel(nn.Module):
    def __init__(self):
        super(SoftLabel, self).__init__()
        self.loss = TotalLoss.soft_label

    def forward(self, stu_logits, tea_logits):
        logits_kl_loss = self.loss(
            f.softmax(stu_logits / self.temperature, dim=1).log(),
            f.softmax(tea_logits / self.temperature, dim=1)
        ) * self.temperature ** 2
        return logits_kl_loss


class ViTKDLoss(nn.Module):
    """PyTorch version of `ViTKD: Practical Guidelines for ViT feature knowledge distillation` """

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 alpha_vitkd=0.00003,
                 beta_vitkd=0.000003,
                 lambda_vitkd=0.5,
                 low_layers_num=2,
                 high_layers_num=1,
                 ):
        super(ViTKDLoss, self).__init__()
        self.alpha_vitkd = alpha_vitkd
        self.beta_vitkd = beta_vitkd
        self.lambda_vitkd = lambda_vitkd

        if student_dims != teacher_dims:
            self.align_low = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(low_layers_num)])
            self.align_high = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(high_layers_num)])
        else:
            self.align_low = None
            self.align_high = None
        self.low_layers_num = low_layers_num
        self.high_layers_num = high_layers_num

        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*2*N*D, B*N*D], student's feature map
            preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
        """
        low_s = preds_S[0]
        low_t = preds_T[0]
        high_s = preds_S[1]
        high_t = preds_T[1]

        B = low_s.shape[0]
        loss_mse = nn.MSELoss(reduction='sum')

        '''ViTKD: Mimicking'''
        if self.align_low is not None:
            if self.low_layers_num == 1:
                low_x = self.align_high(low_s)
            else:
                for i in range(self.low_layers_num):
                    if i == 0:
                        low_x = self.align_low[i](low_s[:, i]).unsqueeze(1)
                    else:
                        low_x = torch.cat((low_x, self.align_low[i](low_s[:, i]).unsqueeze(1)), dim=1)

        else:
            for i in range(self.low_layers_num):
                if i == 0:
                    low_x = low_s[:, i].unsqueeze(1)
                else:
                    low_x = torch.cat((low_x, low_s[:, i].unsqueeze(1)), dim=1)
        xc = low_x

        loss_lr = loss_mse(xc, low_t) / B * self.alpha_vitkd

        '''ViTKD: Generation'''
        # if self.align_high is not None:
        #     if self.high_layers_num == 1:
        #         high_x = self.align_high(high_s)
        #     else:
        #         for i in range(self.high_layers_num):
        #             if i == 0:
        #                 high_x = self.align_high[i](high_s[:, i]).unsqueeze(1)
        #             else:
        #                 high_x = torch.cat((high_x, self.align_high[i](high_s[:, i]).unsqueeze(1)), dim=1)
        #
        # else:
        #     for i in range(self.high_layers_num):
        #         if i == 0:
        #             high_x = high_s[:, i].unsqueeze(1)
        #         else:
        #             high_x = torch.cat((high_x, high_s[:, i].unsqueeze(1)), dim=1)
        # x = high_x
        if self.align_high is not None:
            x = self.align_high(high_s.squeeze())
        else:
            x = high_s.squeeze()
        # Mask tokens
        x = x[:, 1:, :]
        high_t = high_t[:, 0, 1:, :]
        B, N, D = x.shape
        x, mat, ids, ids_masked = self.random_masking(x, self.lambda_vitkd)
        mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
        mask = mat.unsqueeze(-1)

        hw = int(N ** 0.5)

        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
        x = self.generation(x).flatten(2).transpose(1, 2)

        loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
        loss_gen = loss_gen / B * self.beta_vitkd / self.lambda_vitkd

        return loss_lr + loss_gen

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:L]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked


class NKDLoss(nn.Module):
    """PyTorch version of NKD: `Rethinking Knowledge Distillation via Cross-Entropy` """

    def __init__(self,
                 name,
                 use_this,
                 temp=1.0,
                 alpha=1.5,
                 ):
        super(NKDLoss, self).__init__()
        self.temp = temp

    def forward(self, logit_s, logit_t, gt_label):
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        y_i = f.softmax(logit_s, dim=1)
        t_i = f.softmax(logit_t, dim=1)
        # N*1
        y_t = torch.gather(y_i, 1, label)
        w_t = torch.gather(t_i, 1, label).detach()

        mask = torch.zeros_like(logit_s).scatter_(1, label, 1).bool()
        logit_s = logit_s - 1000 * mask
        logit_t = logit_t - 1000 * mask

        # N*class
        T_i = f.softmax(logit_t / self.temp, dim=1)
        S_i = f.softmax(logit_s / self.temp, dim=1)
        # N*1
        T_t = torch.gather(T_i, 1, label)
        S_t = torch.gather(S_i, 1, label)
        # N*class
        np_t = T_i / (1 - T_t)
        np_s = S_i / (1 - S_t)
        np_t[T_i == T_t] = 0
        np_s[T_i == T_t] = 1

        soft_loss = - (w_t * torch.log(y_t)).mean()
        distributed_loss = (np_t * torch.log(np_s)).sum(dim=1).mean()
        distributed_loss = -  (self.temp ** 2) * distributed_loss

        return soft_loss + distributed_loss
