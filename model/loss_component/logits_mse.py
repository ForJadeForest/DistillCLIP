from torch import nn


class LogitsMSE(nn.Module):
    def __init__(self):
        super(LogitsMSE, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, stu_logits, tea_logits):
        return self.loss(stu_logits, tea_logits)
