from model import *
from data import *
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.accelerators import find_usable_cuda_devices
from transformers.optimization import get_cosine_schedule_with_warmup
import os
# os.environ['JSONARGPARSE_DEBUG'] = 'true'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


class MyLightningCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        if lr_scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
        }


cli = MyLightningCLI(seed_everything_default=2022, save_config_overwrite=True,
                     trainer_defaults={'devices': find_usable_cuda_devices(4)})
