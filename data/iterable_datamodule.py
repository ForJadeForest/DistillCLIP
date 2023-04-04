import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.utils import get_dataloader_function, instancialize, load_data_module, load_prepare


class IterableDataModule(pl.LightningDataModule):
    """
    The datModule for webdataset format
    """

    def __init__(self, dataset_para, dataset_file, function_name,
                 val_batch_size, val_num_workers, val_dataset_init_args=None):
        super(IterableDataModule, self).__init__()
        if val_dataset_init_args is None:
            val_dataset_init_args = {}
        self.val_dataset_dict = {}
        self.val_dataset_dict_args = {}
        for dataset_name, dataset_args in val_dataset_init_args.items():
            data_module = load_data_module(dataset_args['dataset_file'], dataset_args['dataset_name'])
            prepare_function = load_prepare(dataset_args['dataset_file'])
            prepare_args = dataset_args.get('prepare_args', None)
            if prepare_args:
                prepare_args.update(dataset_args['dataset_args'])

            self.val_dataset_dict_args[dataset_name] = {
                'data_module': data_module,
                'prepare_function': prepare_function,
                'prepare_args': prepare_args,
                'dataset_args': dataset_args['dataset_args'],
                'is_train': dataset_args['is_train']
            }
        self.get_loader_function = get_dataloader_function(dataset_file, function_name)
        self.dataset_para = dataset_para
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            for dataset_name, init_args in self.val_dataset_dict_args.items():
                self.val_dataset_dict[dataset_name] = instancialize(init_args['data_module'],
                                                                    init_args['dataset_args'],
                                                                    train=init_args['is_train'])

    def train_dataloader(self):
        return self.get_loader_function(**self.dataset_para)

    def val_dataloader(self):
        from pytorch_lightning.utilities import CombinedLoader
        dataloader_dict = {}
        for n, dataset in self.val_dataset_dict.items():
            dataloader_dict[n] = DataLoader(dataset,
                                            batch_size=self.val_batch_size,
                                            num_workers=self.val_num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        return CombinedLoader(iterables=dataloader_dict, mode='max_size')
