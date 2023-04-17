from torch import nn
from torch.nn import functional as f


class OutKLLoss(nn.Module):
    def __init__(self, t=2):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

        self.temperature = t

    def forward(self, stu_out, tea_out):
        return self.loss(
            f.log_softmax(stu_out / self.temperature, dim=1),
            f.softmax(tea_out / self.temperature, dim=1)
        ) * self.temperature ** 2
