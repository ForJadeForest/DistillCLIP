import torch
from torch import nn
from torch.nn import functional as f


class HardLabel(nn.Module):
    def __init__(self):
        super(HardLabel, self).__init__()

    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        return (f.cross_entropy(logits_per_image, labels) + f.cross_entropy(logits_per_text, labels)) / 2
