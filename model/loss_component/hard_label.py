import torch
from torch import nn


class HardLabel(nn.Module):
    def __init__(self):
        super(HardLabel, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.label_cache = None

    def forward(self, stu_logits):
        if self.label_cache is None:
            self.label_cache = torch.arange(stu_logits.shape[0], device=stu_logits.device)

        return self.loss(stu_logits, self.label_cache)
