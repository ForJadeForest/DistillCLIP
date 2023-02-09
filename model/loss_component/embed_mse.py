from torch import nn


class EmbedMSELoss(nn.Module):
    def __init__(self):
        super(EmbedMSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, stu_embedding, tea_embedding):
        return self.loss(stu_embedding, tea_embedding)
