from pytorch_lightning.callbacks import TQDMProgressBar

from model import *
from data import *
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.accelerators import find_usable_cuda_devices
import os

# os.environ['JSONARGPARSE_DEBUG'] = 'true'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

device_num = int(os.getenv('DEVICE_NUM', 4))

torch.set_float32_matmul_precision('high')

class IterableDatasetProgressBar(TQDMProgressBar):
    def __init__(self, total_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = total_length

    def init_train_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.total = self.total
        return bar


cli = LightningCLI(seed_everything_default=2022, save_config_callback=None,
                   trainer_defaults={'devices': find_usable_cuda_devices(2)})
