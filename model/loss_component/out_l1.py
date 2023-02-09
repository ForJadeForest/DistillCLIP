from torch import nn


class OutL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, stu_out, tea_out):
        return self.loss(stu_out, tea_out)
