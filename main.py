from model import *
from data import *
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import cli as pl_cli
from transformers.optimization import get_cosine_schedule_with_warmup
#
#
# @pl_cli.LR_SCHEDULER_REGISTRY
# def warm_up_scheduler(*args, **kwargs):
#     return get_cosine_schedule_with_warmup(*args, **kwargs)


class MyLightningCLI(pl_cli.LightningCLI):
    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        if lr_scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
        }


cli = MyLightningCLI(seed_everything_default=2022, save_config_overwrite=True)
