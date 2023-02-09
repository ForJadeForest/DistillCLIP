import torch
from torch import nn


class SMDMultiModel(nn.Module):
    def __init__(self, tau=0.04, topk=1):
        super(SMDMultiModel, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.tau = tau
        self.topk = topk

    def forward(self, teacher_inputs, inputs, text_inputs, normalized=True):
        n = inputs.size(0)

        if normalized:
            inputs = torch.nn.functional.normalize(inputs, dim=1)
            teacher_inputs = torch.nn.functional.normalize(teacher_inputs, dim=1)
        # Teacher 的分布中，每一个样本对应的距离
        x1 = torch.pow(teacher_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_t = x1 + x1.t()
        dist_t.addmm_(teacher_inputs, teacher_inputs.t(), beta=1, alpha=-2)
        dist_t = dist_t.clamp(min=1e-12).sqrt()  # for numerical stability

        # Compute pairwise distance
        # Teacher 与 Student 的对应距离
        x1 = torch.pow(teacher_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        x2 = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = x1 + x2.t()
        dist.addmm_(teacher_inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # Compute image text distance
        x1 = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        x2 = torch.pow(text_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_text = x1 + x2.t()
        dist_text.addmm_(teacher_inputs, inputs.t(), beta=1, alpha=-2)
        dist_text = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        text_positive = dist_text.diag()

        # For each anchor, find the hardest positive and negative
        negative_index = (dist_t > torch.diag(dist).expand(n, n).t()).float()
        negative = dist * negative_index
        negative[negative_index == 0] = 1e5
        positive_index = 1 - negative_index
        positive = dist * positive_index

        dist_an = torch.min(negative, dim=1)
        dist_ap = torch.max(positive, dim=1)

        an_t = torch.gather(dist_t, 1, dist_an.indices.unsqueeze(1)).squeeze()
        ap_t = torch.gather(dist_t, 1, dist_ap.indices.unsqueeze(1)).squeeze()

        weight_an = torch.clamp_min(an_t.detach() - dist_an.values.detach(), min=0.)
        weight_ap = torch.clamp_min(dist_ap.values.detach() - ap_t.detach(), min=0.)

        weight_dist_an = weight_an * dist_an.values / self.tau
        weight_dist_ap = weight_ap * dist_ap.values / self.tau
        weight_dist_text_positive = 1 * text_positive / self.tau

        logits = torch.cat([weight_dist_an.unsqueeze(-1),
                            weight_dist_ap.unsqueeze(-1),
                            weight_dist_text_positive.unspueeze(-1)], dim=1)
        labels = torch.zeros(weight_dist_an.shape[0], dtype=torch.long).cuda()

        return self.loss(logits, labels)
