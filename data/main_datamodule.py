import importlib
import inspect

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class MainDataModule(pl.LightningDataModule):
    def __init__(self, dataset_para, dataset, dataset_name,
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
        self.load_data_module()
        self.save_hyperparameters()

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

    def load_data_module(self):
        dataset_file = self.dataset
        name = self.dataset_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        try:
            # importlib.import_module('.xxx_dataset')动态的导入了模块
            # getattr 是获取属性值，获取模块中 XxxDataset 这一类的属性
            self.data_module = getattr(importlib.import_module(
                ".component." + dataset_file, package=__package__), name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset_file}.{name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.dataset_para.
        """
        # 获取self.data_module (self.dataset对应的类) 的__init__函数的参数
        class_args = inspect.signature(self.data_module.__init__).parameters

        inkeys = self.dataset_para.keys()
        args1 = {}
        for arg in class_args:
            # 如果需要的参数在kwargs的key中，则赋值
            if arg in inkeys:
                args1[arg] = self.dataset_para[arg]
        args1.update(other_args)
        return self.data_module(**args1)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--dataset', default='_dataset', type=str)
    parser.add_argument('--dataset_name', default='TextDataset', type=str)
    parser.add_argument('--data_dir', default='/data/pyz/data', type=str)
    args = parser.parse_args()
    data_module = DistillationDataModule(**vars(args))
    data_module.setup()
