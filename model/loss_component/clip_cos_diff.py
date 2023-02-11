import torch
from torch import nn


def get_neg_element(x):
    m, n = x.shape
    assert m == n
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CLIPCosDiff(nn.Module):
    def __init__(self):
        super(CLIPCosDiff, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, stu_logits, tea_logits):
        stu_pos_dis = torch.diagonal(stu_logits)
        tea_pos_dis = torch.diagonal(tea_logits)
        pos_loss = torch.sum(self.relu(stu_pos_dis - tea_pos_dis))
        stu_neg_dis = self.get_neg_element(stu_logits)
        tea_neg_dis = self.get_neg_element(tea_logits)
        neg_loss = torch.sum(self.relu(tea_neg_dis - stu_neg_dis))
        return neg_loss + pos_loss
