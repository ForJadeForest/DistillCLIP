from torch import nn
from torch.nn import functional as f


class LastValueMapKL(nn.Module):
    def __init__(self):
        super(LastValueMapKL, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, stu_value_map, tea_value_map):
        return self.loss(
            f.softmax(stu_value_map, dim=1).log(),
            f.softmax(tea_value_map, dim=1)
        )
