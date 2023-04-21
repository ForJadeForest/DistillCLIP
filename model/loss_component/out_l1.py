from torch import nn
from typing import  Optional, List

class OutL1Loss(nn.Module):
    def __init__(self, dim_map: Optional[List] = None):
        super().__init__()
        self.loss = nn.L1Loss()
        self.aline_linear = None
        if dim_map:
            self.aline_linear = nn.Linear(dim_map[0], dim_map[1])

    def forward(self, stu_out, tea_out):
        if self.aline_linear:
            stu_out = self.aline_linear(stu_out)
        return self.loss(stu_out, tea_out)
