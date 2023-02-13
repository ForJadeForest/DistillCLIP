from torch import nn


class HiddenMSE(nn.Module):
    def __init__(self):
        super(HiddenMSE, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, stu_hidden, tea_hidden):
        res_loss = 0
        for layer_num, (stu_out, tea_out) in enumerate(zip(stu_hidden, tea_hidden)):
            if layer_num == 0:
                res_loss = self.loss(stu_out, tea_out)
            else:
                res_loss += self.loss(stu_out, tea_out)
        res_loss /= len(stu_hidden)
        return res_loss
