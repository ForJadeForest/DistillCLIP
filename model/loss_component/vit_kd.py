import torch
from torch import nn


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_masked = ids_shuffle[:, len_keep:L]

    x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_keep, mask, ids_restore, ids_masked


class ViTKDLoss(nn.Module):
    """PyTorch version of `ViTKD: Practical Guidelines for ViT feature knowledge distillation` """

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 alpha_vitkd=0.00003,
                 beta_vitkd=0.000003,
                 lambda_vitkd=0.5,
                 low_layers_num=2,
                 high_layers_num=1,
                 ):
        super(ViTKDLoss, self).__init__()
        self.alpha_vitkd = alpha_vitkd
        self.beta_vitkd = beta_vitkd
        self.lambda_vitkd = lambda_vitkd

        if student_dims != teacher_dims:
            self.align_low = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(low_layers_num)])
            self.align_high = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(high_layers_num)])
        else:
            self.align_low = None
            self.align_high = None
        self.low_layers_num = low_layers_num
        self.high_layers_num = high_layers_num

        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*2*N*D, B*N*D], student's feature map
            preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
        """
        low_s = preds_S[0]
        low_t = preds_T[0]
        high_s = preds_S[1]
        high_t = preds_T[1]

        B = low_s.shape[0]
        loss_mse = nn.MSELoss(reduction='sum')

        '''ViTKD: Mimicking'''
        low_x = None
        for i in range(self.low_layers_num):
            low_align_rep = low_s[:, i]
            if self.align_low:
                low_align_rep = self.align_low[i](low_s[:, i])
            low_align_rep = low_align_rep.unsqueeze(1)
            if i == 0:
                low_x = low_align_rep
            else:
                low_x = torch.cat((low_x, low_align_rep), dim=1)

        loss_lr = loss_mse(low_x, low_t) / B * self.alpha_vitkd

        '''ViTKD: Generation'''
        loss_gen = 0
        for i in range(self.high_layers_num):
            align_layer = None
            if self.align_high:
                align_layer = self.align_high[i]
            if i == 0:
                loss_gen = self.generation_loss(high_s[:, i], align_layer, high_t[:, i])
            else:
                loss_gen += self.generation_loss(high_s[:, i], align_layer, high_t[:, i])
        loss_gen /= self.high_layers_num
        return loss_lr + loss_gen

    def generation_loss(self, high_layer_output, align_linear, tea_output):
        loss_mse = nn.MSELoss(reduction='sum')
        if self.align_high is not None:
            high_layer_output = align_linear(high_layer_output)

        x = high_layer_output
        x = x[:, 1:, :]
        tea_output = tea_output[:, 1:, :]
        B, N, D = x.shape
        x, mat, ids, ids_masked = random_masking(x, self.lambda_vitkd)
        mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
        mask = mat.unsqueeze(-1)

        hw = int(N ** 0.5)

        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
        x = self.generation(x).flatten(2).transpose(1, 2)

        loss_gen = loss_mse(torch.mul(x, mask), torch.mul(tea_output, mask))
        loss_gen = loss_gen / B * self.beta_vitkd / self.lambda_vitkd
        return loss_gen
