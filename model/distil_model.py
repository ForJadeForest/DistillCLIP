import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import cli as pl_cli
from torch import nn, optim
from torch.nn import functional as f
from torchmetrics.functional import accuracy
try:
    from _utils import load, build_model, get_transformer_para, teacher_load
    from _common import VisionTransformer
    from _text_encoder import TextEncoder
except ModuleNotFoundError:
    from ._utils import load, build_model, get_transformer_para, teacher_load
    from ._common import VisionTransformer
    from ._text_encoder import TextEncoder


# 导入需要的组件


# from _utils import xxx
# from _metrics import xxx
def layer_map(stu_layer_num, step):
    return stu_layer_num * step


@pl_cli.MODEL_REGISTRY
class DistillModel(pl.LightningModule):
    def __init__(self, student_encoder: nn.Module, teacher_name, context_length, t, download_root, loss_weight,
                 model_type='text'):
        super().__init__()
        # self.example_input_array = torch.tensor((torch.randint(low=0, high=300, size=(64, 77))))
        self.__dict__.update(locals())
        self.save_hyperparameters(ignore=['student_encoder'])
        # 定义模型
        self.student = student_encoder
        self.teacher_name = teacher_name
        self.teacher = teacher_load(teacher_name, download_root=download_root)
        for p in self.teacher.parameters():
            p.requires_grad = False
        # 定义指标
        self.loss_mse = nn.MSELoss()
        self.loss_kl = loss_function = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input):
        student_outs = self.student(input, only_last_state=False)
        teacher_outs = self.teacher(input, only_last_state=False)
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
        assert tea_layer_num % stu_layer_num == 0
        step = tea_layer_num // stu_layer_num
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

        loss = 0
        for w, l in zip(self.hparams.loss_weight, [emb_loss, attn_loss, hidden_loss, logits_loss]):
            loss += w * l

        return loss, emb_loss, attn_loss, hidden_loss, logits_loss

    def training_step(self, inputs, batch_idx):
        # self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)
        loss, emb_loss, attn_loss, hidden_loss, logits_loss = self.cal_loss(student_outs, teacher_outs)

        # Logging to TensorBoard by default
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/emb_loss", emb_loss, on_step=True)
        self.log("train/attn_loss", attn_loss, on_step=True)
        self.log("train/hidden_loss", hidden_loss, on_step=True)
        self.log("train/logits_loss", logits_loss, on_step=True)

        return loss

    def validation_step(self, inputs, batch_idx):
        imgs, texts, captions = inputs
        student_outs, teacher_outs = self.forward(texts)
        loss, emb_loss, attn_loss, hidden_loss, logits_loss = self.cal_loss(student_outs, teacher_outs)
        label = torch.arange(student_outs[0].shape[0], device=self.device)
        stu_logits, tea_logits = norm_and_logits(imgs, student_outs[0], teacher_outs[0])[:2]
        k_list = [i for i in [1, 2, 3, 4, 5, 10, 20, 30, 50] if i < len(imgs)]
        for k in k_list:
            acc = accuracy(stu_logits, label, top_k=k)
            acc_tea = accuracy(tea_logits, label, top_k=k)
            self.log('val/acc_top{}'.format(k), acc, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)
            self.log('val/tea_acc_top{}'.format(k), acc_tea, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)
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

def norm_and_logits(image_encode, stu_text_encode, tea_text_encode):
    image_encode = image_encode / image_encode.norm(dim=1, keepdim=True)
    stu_text_encode = stu_text_encode / stu_text_encode.norm(dim=1, keepdim=True)
    tea_text_encode = tea_text_encode / tea_text_encode.norm(dim=1, keepdim=True)
    stu_text_logits = stu_text_encode @ image_encode.t()
    tea_text_logits = tea_text_encode @ image_encode.t()
    return stu_text_logits, tea_text_logits, stu_text_logits.T, tea_text_logits.T
