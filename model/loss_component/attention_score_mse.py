import torch
from torch import nn


class AttentionScoreMSE(nn.Module):
    def __init__(self):
        super(AttentionScoreMSE, self).__init__()
        self.loss = nn.MSELoss()

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
