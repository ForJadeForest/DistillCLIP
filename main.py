from data import *
from model import *
from pytorch_lightning.utilities import cli as pl_cli
import torch
import pytorch_lightning as pl


# 导入torch中所有的优化器
pl_cli.OPTIMIZER_REGISTRY.register_classes(module=torch.optim, base_cls=torch.optim.Optimizer)
pl_cli.CALLBACK_REGISTRY.register_classes(module=pl.callbacks, base_cls=pl.Callback)
cli = pl_cli.LightningCLI(seed_everything_default=2022)
