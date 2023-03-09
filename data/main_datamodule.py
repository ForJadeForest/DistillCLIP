import importlib
import inspect

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class MainDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_para,
                 dataset,
                 dataset_name,
                 prepare_para=None,
                 num_workers=8,
                 train_batch_size=128,
                 val_batch_size=1250):
        """
         dataset_para: The dataset parameters
         dataset: The *.py file name of the dataset class
         dataset_name: The dataset Class name
         """
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.dataset_para = dataset_para
        self.dataset_name = dataset_name

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.data_module = self.load_data_module()

        self.prepare_function = self.load_prepare()
        self.prepare_function_args = prepare_para
        if self.prepare_function_args:
            self.prepare_function_args.update(dataset_para)

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if self.prepare_function:
            self.prepare_function(self.prepare_function_args)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(train=True)
            self.valset = self.instancialize(train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(train=False)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False)

    def load_prepare(self):
        dataset_file = self.dataset
        module = importlib.import_module(".component." + dataset_file, package=__package__)
        if hasattr(module, 'prepare'):
            prepare_function = getattr(module, 'prepare')
        else:
            prepare_function = None
        return prepare_function

    def load_data_module(self):
        dataset_file = self.dataset
        name = self.dataset_name
        try:
            data_module = getattr(importlib.import_module(
                ".component." + dataset_file, package=__package__), name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset_file}.{name}')
        return data_module

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.dataset_para.
        """
        # get self.data_module (the self.dataset class) __init__ function parameters
        class_args = inspect.signature(self.data_module.__init__).parameters

        inkeys = self.dataset_para.keys()
        args1 = {}
        for arg in class_args:
            # if the args in inkeys
            if arg in inkeys:
                args1[arg] = self.dataset_para[arg]
        args1.update(other_args)
        return self.data_module(**args1)
