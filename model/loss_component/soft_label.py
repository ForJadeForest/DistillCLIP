from torch import nn
from torch.nn import functional as f


class SoftLabel(nn.Module):
    def __init__(self, temperature=2):
        super(SoftLabel, self).__init__()
        self.temperature = temperature

    def forward(self, stu_logits, tea_logits):
        def dist_loss(teacher_logits, student_logits):
            return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
        return dist_loss(tea_logits / self.temperature, stu_logits / self.temperature) * self.temperature ** 2
