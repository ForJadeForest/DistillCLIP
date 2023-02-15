import torch
from torch import nn


class OutCosLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, stu_out: torch.Tensor, tea_out):
        return self.loss(stu_out, tea_out, torch.ones(len(stu_out), device=stu_out.device))
