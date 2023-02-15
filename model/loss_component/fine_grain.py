import torch
from torch import nn
from torch.nn import functional as f


class FineGrainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, image_out, text_out):
        def cal_similarity(query_features, respond_features):
            """
            query_features: [B, n1, d]
            respond_features: [B, n2, d]
            """
            res = []
            for q in query_features:
                # similarity for q to all respond_features: [B, n1, n2]
                similarity = torch.matmul(q, respond_features.permute(0, 2, 1))

                max_res = similarity.max(dim=-1).values
                # max_res: [B, n1]
                mean_res = max_res.mean(dim=-1)
                # mean_res: [B,]
                res.append(mean_res)
            similarity_total = torch.stack(res, dim=0)  # [B, B]
            return similarity_total

        i2t_similarity = cal_similarity(image_out, text_out)
        t2i_similarity = cal_similarity(text_out, image_out)

        label = torch.arange(i2t_similarity.shape[0], device=i2t_similarity.device)
        return 0.5 * (self.loss(i2t_similarity, label) + self.loss(t2i_similarity, label))
