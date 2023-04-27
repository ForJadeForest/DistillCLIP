from torch import nn


class SoftLabel(nn.Module):
    def __init__(self):
        super(SoftLabel, self).__init__()

    def forward(self, stu_logits_per_image, stu_logits_per_text,
                tea_logits_per_image, tea_logits_per_text):
        def dist_loss(teacher_logits, student_logits):
            return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

        loss = dist_loss(tea_logits_per_image, stu_logits_per_image) + \
               dist_loss(tea_logits_per_text, stu_logits_per_text)
        return loss / 2
