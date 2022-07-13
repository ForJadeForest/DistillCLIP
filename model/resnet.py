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


@pl_cli.MODEL_REGISTRY
class Resnet(pl.LightningModule):
    def __init__(self, in_channel=3, lr=1e-3, class_num=10):
        super().__init__()
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.__dict__.update(locals())
        self.save_hyperparameters()
        # 定义模型
        self.t = nn.Conv2d(in_channel, 3, kernel_size=3)
        self.encoder = resnet18()
        self.linear = nn.Linear(1000, class_num)
        # 定义指标
        self.train_metric = Accuracy(num_classes=class_num)
        self.valid_metric = Accuracy(num_classes=class_num)

    def forward(self, x):
        x = self.t(x)
        x = self.encoder(x)
        x = f.softmax(self.linear(x), dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = self.t(x)
        # x = self.encoder(x)
        # x = f.softmax(self.linear(x), dim=1)
        x = self.forward(x)
        self.train_metric(torch.argmax(x, dim=1), y)
        loss = f.cross_entropy(x, y)
        # Logging to TensorBoard by default
        self.log("train/loss", loss, on_epoch=True)
        # log step 和 epoch
        self.log('train/acc', self.train_metric, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.t(x)
        x = self.encoder(x)
        x = f.softmax(self.linear(x), dim=1)
        self.valid_metric.update(torch.argmax(x, dim=1), y)
        loss = f.cross_entropy(x, y)
        self.log("valid/loss", loss)
        self.log("hp_metric", self.valid_metric)
        self.log('valid/acc', self.valid_metric)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

        return optimizer, scheduler
