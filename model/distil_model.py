import pytorch_lightning as pl
from torch import nn, optim, utils
from torchvision.models import resnet18
from torch.nn import functional as f
from pytorch_lightning.utilities import cli as pl_cli
from torchmetrics import Accuracy
import torch


# 导入需要的组件
# from _common import xxx
# from _utils import xxx
# from _metrics import xxx
def layer_map(stu_layer_num, step):
    return stu_layer_num * step


@pl_cli.MODEL_REGISTRY
class DistillModel(pl.LightningModule):
    def __init__(self, student_encoder, teacher_encoder, t):
        super().__init__()
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.__dict__.update(locals())
        self.save_hyperparameters()
        # 定义模型
        self.student = student_encoder
        self.teacher = teacher_encoder
        # 定义指标

        self.loss_mse = nn.MSELoss()
        self.loss_kl = loss_function = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input):
        student_outs = self.student(input)
        teacher_outs = self.teacher(input)
        return student_outs, teacher_outs

    def cal_loss(self, student_outs, teacher_outs):
        stu_last_rep, stu_attention_maps, stu_representations, stu_embedding = student_outs
        tea_last_rep, tea_attention_maps, tea_representations, tea_embedding = teacher_outs

        # calculate the embedding loss
        emb_loss = self.loss_mse(stu_embedding, tea_embedding)

        # calculate the middle layers loss
        stu_layer_num = len(stu_attention_maps)
        tea_layer_num = len(tea_attention_maps)
        # attention loss
        attn_loss = 0
        step = tea_layer_num / stu_layer_num
        for layer_num, stu_attn_out in enumerate(stu_attention_maps):
            attn_loss += self.loss_mse(stu_attention_maps[layer_num], tea_attention_maps[layer_map(layer_num, step)])
        attn_loss /= stu_layer_num
        hidden_loss = 0
        for layer_num, stu_attn_out in enumerate(stu_representations):
            hidden_loss += self.loss_mse(stu_representations[layer_num],
                                         tea_representations[layer_map(layer_num, step)])
        hidden_loss /= stu_layer_num

        # calculate the pred loss
        logits_loss = self.loss_kl(
            f.softmax(stu_last_rep / self.hparams.t, dim=1).log(),
            f.softmax(tea_last_rep / self.hparams.t, dim=1)
        ) * self.hparams.t ** 2

        loss = emb_loss + attn_loss + hidden_loss + logits_loss

        return loss, emb_loss, attn_loss, hidden_loss, logits_loss

    def training_step(self, inputs, batch_idx):
        student_outs, teacher_outs = self.forward(inputs)
        loss, emb_loss, attn_loss, hidden_loss, logits_loss = self.cal_loss(student_outs, teacher_outs)

        # Logging to TensorBoard by default
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/emb_loss", emb_loss, on_step=True)
        self.log("train/attn_loss", attn_loss, on_step=True)
        self.log("train/hidden_loss", hidden_loss, on_step=True)
        self.log("train/logits_loss", logits_loss, on_step=True)

        # log step 和 epoch
        return loss

    def validation_step(self, inputs, batch_idx):
        student_outs, teacher_outs = self.forward(inputs)
        loss, emb_loss, attn_loss, hidden_loss, logits_loss = self.cal_loss(student_outs, teacher_outs)

        # Logging to TensorBoard by default
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/emb_loss", emb_loss, on_step=True)
        self.log("val/attn_loss", attn_loss, on_step=True)
        self.log("val/hidden_loss", hidden_loss, on_step=True)
        self.log("val/logits_loss", logits_loss, on_step=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

        return optimizer, scheduler
