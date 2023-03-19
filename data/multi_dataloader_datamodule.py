import pytorch_lightning as pl
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

from data.utils import load_prepare, load_data_module, instancialize, SequentialLoader

"""
dataset_args:{
    train: {
        dataset_args1, dataset_args2
    }
}

init_dataset:{
    dataset1: {
        is_train: True,
        dataset_file: xxxx,
        dataset_name: ImageDataset,
        dataset_args:{
            args1: xxx,
            args2: xxx
        }
        prepare_args: {
            args1: xxx,
            args2: xxx   
        }
    },
    dataset2: {
        
    }
}

"""


class MultiDataloaderMainDataModule(pl.LightningDataModule):
    def __init__(self,
                 init_dataset_args,
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
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.data_module_list = []
        self.prepare_function_list = []
        self.prepare_args_list = []
        self.dataset_args_list = []
        self.stage_list = []
        self.dataset_dict = {}
        for dataset_name, dataset_args in init_dataset_args.items():
            data_module = load_data_module(dataset_args['dataset_file'], dataset_args['dataset_name'])
            prepare_function = load_prepare(dataset_args['dataset_file'])
            prepare_args = dataset_args.get('prepare_args', None)
            if prepare_args:
                prepare_args.update(dataset_args['dataset_args'])

            self.dataset_dict[dataset_name] = {
                'data_module': data_module,
                'prepare_function': prepare_function,
                'prepare_args': prepare_args,
                'dataset_args': dataset_args['dataset_args'],
                'is_train': dataset_args['is_train']
            }

        self.save_hyperparameters()
        self.train_dataset_dict = {}
        self.val_dataset_dict = {}

    def prepare_data(self) -> None:
        for prepare_function, prepare_args in zip(self.prepare_function_list, self.prepare_args_list):
            if prepare_function:
                prepare_function(**prepare_args)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            for dataset_name, init_args in self.dataset_dict.items():
                if init_args['is_train']:
                    self.train_dataset_dict[dataset_name] = instancialize(init_args['data_module'],
                                                                          init_args['dataset_args'],
                                                                          train=init_args['is_train'])
                else:
                    self.val_dataset_dict[dataset_name] = instancialize(init_args['data_module'],
                                                                        init_args['dataset_args'],
                                                                        train=init_args['is_train'])

    def train_dataloader(self):
        dataloader_dict = {}
        for n, dataset in self.train_dataset_dict.items():
            dataloader_dict[n] = DataLoader(dataset,
                                            batch_size=self.train_batch_size,
                                            num_workers=self.num_workers,
                                            shuffle=False,
                                            pin_memory=True)

        return SequentialLoader(*dataloader_dict.values())

    def val_dataloader(self):
        dataloader_dict = {}
        for n, dataset in self.val_dataset_dict.items():
            dataloader_dict[n] = DataLoader(dataset,
                                            batch_size=self.val_batch_size,
                                            num_workers=self.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        return CombinedLoader(iterables=dataloader_dict, mode='max_size')
