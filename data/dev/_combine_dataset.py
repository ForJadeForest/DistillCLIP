from typing import List

import numpy as np
from torch.utils.data import Dataset


class CombineDataset(Dataset):
    def __init__(self, dataset_list: List[Dataset], train: bool = True):

        self.datasets = dataset_list
        self.lengths = [len(dataset) for dataset in dataset_list]
        self.cum_lengths = np.array([0] + self.lengths).cumsum()
        self.combine_length = sum(self.lengths)
        self.is_train = train
        for dataset in self.datasets:
            # assert hasattr(dataset, 'check_mode')
            dataset.check_mode(train)

    def __len__(self):
        return self.combine_length

    def map_idx(self, idx):
        dataset_idx = 1
        while idx >= self.cum_lengths[dataset_idx]:
            dataset_idx += 1
        return dataset_idx - 1, idx - self.cum_lengths[dataset_idx - 1]

    def __getitem__(self, idx):
        dataset_idx, new_idx = self.map_idx(idx)
        return self.datasets[dataset_idx][new_idx]


if __name__ == '__main__':
    class TestDataset(Dataset):
        def __init__(self, begin, end):
            self.data = [i for i in range(begin, end)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            return self.data[item]


    datasets = [TestDataset(b, e + 1) for b, e in zip([1, 4, 6, 8, 12], [3, 5, 7, 11])]
    for d in datasets:
        print(d.data)
    com_datasets = CombineDataset(datasets)
    for i in range(0, len(com_datasets)):
        print(com_datasets[i], i)
