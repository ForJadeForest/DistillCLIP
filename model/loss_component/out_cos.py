import torch
from typing import List, Optional
from torch import nn


class OutCosLoss(nn.Module):
    def __init__(self, dim_map: Optional[List] = None):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()
        self.aline_linear = None
        if dim_map:
            self.aline_linear = nn.Linear(dim_map[0], dim_map[1])

    def forward(self, stu_out: torch.Tensor, tea_out: torch.Tensor):
        if self.aline_linear:
            stu_out = self.aline_linear(stu_out)
        return self.loss(stu_out, tea_out, torch.ones(len(stu_out), device=stu_out.device))
