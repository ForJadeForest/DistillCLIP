from model import *
from data import *
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import cli as pl_cli

# 导入torch中所有的优化器
pl_cli.OPTIMIZER_REGISTRY.register_classes(module=torch.optim, base_cls=torch.optim.Optimizer)
pl_cli.CALLBACK_REGISTRY.register_classes(module=pl.callbacks, base_cls=pl.Callback)
cli = pl_cli.LightningCLI(seed_everything_default=2022, save_config_overwrite=True)
