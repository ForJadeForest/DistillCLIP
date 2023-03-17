import importlib
import inspect
from torch.utils.data import DataLoader


def load_data_module(dataset_file, dataset_name):
    try:
        data_module = getattr(importlib.import_module(
            ".component." + dataset_file, package=__package__), dataset_name)
    except:
        raise ValueError(
            f'Invalid Dataset File Name or Invalid Class Name data.{dataset_file}.{dataset_name}')
    return data_module


def load_prepare(dataset_file):
    module = importlib.import_module(".component." + dataset_file, package=__package__)
    if hasattr(module, 'prepare'):
        prepare_function = getattr(module, 'prepare')
    else:
        prepare_function = None
    return prepare_function


def instancialize(data_module, dataset_args, **other_args):
    """ Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.dataset_para.
    """
    # get self.data_module (the self.dataset class) __init__ function parameters
    class_args = inspect.signature(data_module.__init__).parameters

    inkeys = dataset_args.keys()
    args1 = {}
    for arg in class_args:
        # if the args in inkeys
        if arg in inkeys:
            args1[arg] = dataset_args[arg]
    args1.update(other_args)
    return data_module(**args1)


class SequentialLoader:
    def __init__(self, *dataloaders: DataLoader):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader
