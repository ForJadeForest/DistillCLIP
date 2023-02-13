from torch import nn


class OutCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, stu_out, tea_out):
        return self.loss(
            stu_out,  # [batch, out_dim]
            tea_out.softmax(dim=1)
        )
