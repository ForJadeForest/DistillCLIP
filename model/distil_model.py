import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import cli as pl_cli
from torch import nn, optim
from torch.nn import functional as f
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

try:
    from _utils import teacher_load, LayerMap
except ModuleNotFoundError:
    from ._utils import teacher_load, LayerMap


@pl_cli.MODEL_REGISTRY
class DistillModel(pl.LightningModule):
    def __init__(self, student_encoder: nn.Module, teacher_name, t, download_root, loss_weight,
                 model_type='text', lr=1e-3, map_type='mid'):
        super().__init__()
        # self.example_input_array = torch.tensor((torch.randint(low=0, high=300, size=(64, 77))))
        self.__dict__.update(locals())
        self.save_hyperparameters(ignore=['student_encoder'])
        # 定义模型
        self.student = student_encoder
        self.teacher_name = teacher_name
        self.teacher, tea_layer_num = teacher_load(teacher_name, download_root, model_type)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.layer_map = LayerMap(student_encoder.layers, tea_layer_num, map_type)
        # 定义指标
        self.loss_mse = nn.MSELoss()
        self.loss_kl = loss_function = nn.KLDivLoss(reduction='batchmean')
        self.k_list = [i for i in [1, 2, 3, 4, 5, 10, 20, 30, 50]]
        self.acc_metrics = []
        for k in self.k_list:
            self.acc_metrics.append(Accuracy(top_k=k))

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
            attn_loss += self.loss_mse(stu_attention_maps[layer_num], tea_attention_maps[self.layer_map(layer_num)])
        attn_loss /= stu_layer_num
        hidden_loss = 0
        for layer_num, stu_attn_out in enumerate(stu_representations):
            hidden_loss += self.loss_mse(stu_representations[layer_num],
                                         tea_representations[self.layer_map(layer_num)])
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
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)
        loss, emb_loss, attn_loss, hidden_loss, logits_loss = self.cal_loss(student_outs, teacher_outs)

        # Logging to TensorBoard by default
        self.log_info('train', loss, emb_loss, attn_loss, hidden_loss, logits_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, texts, sentence = batch
        if self.hparams.model_type == 'text':
            inputs, contrary = texts, imgs
            student_outs, teacher_outs = self.forward(inputs)
            loss, emb_loss, attn_loss, hidden_loss, logits_loss = self.cal_loss(student_outs, teacher_outs)
            label = torch.arange(student_outs[0].shape[0], device=self.device)
            stu_logits, tea_logits = norm_and_logits(imgs, student_outs[0], teacher_outs[0], type='text')[:2]
        else:
            inputs, contrary = imgs, texts
            student_outs, teacher_outs = self.forward(inputs)
            loss, emb_loss, attn_loss, hidden_loss, logits_loss = self.cal_loss(student_outs, teacher_outs)
            label = torch.arange(student_outs[0].shape[0], device=self.device)
            from clip import load
            clip_model, _ = load(self.teacher_name, device=self.device, download_root=self.hparams.download_root)
            tea_text_logits = clip_model.encode_text(contrary)
            stu_logits, tea_logits = norm_and_logits(tea_text_logits, student_outs[0], teacher_outs[0], type='image')[
                                     :2]
        self.log('hp_metric', self.acc_metrics[0], metric_attribute='acc_metrics', batch_size=len(inputs))
        for i, metric in enumerate(self.acc_metrics):
            metric.to(self.device)
            metric(stu_logits, label)

            self.log('hp_metric/stu_acc_top{}'.format(self.k_list[i]), metric, metric_attribute='acc_metrics',
                     batch_size=len(inputs))
            if self.current_epoch == 0:
                acc_tea = accuracy(tea_logits, label, top_k=self.k_list[i])
                self.log('hp_metric/tea_acc_top{}'.format(self.k_list[i]), acc_tea, prog_bar=False, sync_dist=True,
                         batch_size=len(inputs))
        # Logging to TensorBoard by default
        self.log_info('val', loss, emb_loss, attn_loss, hidden_loss, logits_loss)
        return loss

    def log_info(self, stage, loss, emb_loss, attn_loss, hidden_loss, logits_loss):
        self.log("{}/loss".format(stage), loss, on_epoch=True, on_step=True)
        self.log("{}/emb_loss".format(stage), emb_loss, on_epoch=True, on_step=True)
        self.log("{}/attn_loss".format(stage), attn_loss, on_epoch=True, on_step=True)
        self.log("{}/hidden_loss".format(stage), hidden_loss, on_epoch=True, on_step=True)
        self.log("{}/logits_loss".format(stage), logits_loss, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

        return optimizer, scheduler


def norm_and_logits(encode, stu_encode, tea_encode, type):
    if type == 'text':
        image_encode = encode
        image_encode = image_encode / image_encode.norm(dim=1, keepdim=True)
        stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
        tea_encode = tea_encode / tea_encode.norm(dim=1, keepdim=True)
        stu_logits = stu_encode @ image_encode.t()
        tea_logits = tea_encode @ image_encode.t()
        return stu_logits, tea_logits, stu_logits.T, tea_logits.T
    else:
        text_encode = encode
        text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
        stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
        tea_encode = tea_encode / tea_encode.norm(dim=1, keepdim=True)
        stu_logits = text_encode @ stu_encode.t()
        tea_logits = text_encode @ tea_encode.t()
        return stu_logits, tea_logits, stu_logits.T, tea_logits.T
