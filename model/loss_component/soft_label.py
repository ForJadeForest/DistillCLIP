from torch import nn
from torch.nn import functional as f


class SoftLabel(nn.Module):
    def __init__(self, temperature):
        super(SoftLabel, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')
        self.temperature = temperature

    def forward(self, stu_logits, tea_logits):
        logits_kl_loss = self.loss(
            f.softmax(stu_logits / self.temperature, dim=1).log(),
            f.softmax(tea_logits / self.temperature, dim=1)
        ) * self.temperature ** 2
        return logits_kl_loss
