from pathlib import Path

import pytorch_lightning as pl
import webdataset as wds
from clip import tokenize
from sklearn.model_selection import train_test_split
from torchvision import transforms

from .component.utils import IMAGE_MEAN, IMAGE_STD


class TextImageDataModule(pl.LightningDataModule):
    """
    The datModule for webdataset format
    """
    def __init__(self, image_path, batch_size=64, num_workers=4):
        super(TextImageDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        print("batch_size", self.batch_size, "num_workers", self.num_workers)
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        url = [str(i) for i in list(Path(image_path).glob('*.tar'))]
        self.train_url, self.val_url = train_test_split(url, test_size=0.05)
        print(f'len(train) == {len(self.train_url)}, len(val) == {len(self.val_url)}')

    def make_transform(self, is_train):
        if is_train:
            return transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.RandAugment(num_ops=4),
                    transforms.ToTensor(),
                    transforms.Normalize(self.img_mean, self.img_std),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.img_mean, self.img_std)
                ]
            )

    def make_loader(self, is_train):
        if is_train:
            urls = self.train_url
            shuffle = 5000
        else:
            urls = self.val_url
            shuffle = 0

        transform = self.make_transform(is_train)

        dataset = (
            wds.WebDataset(urls, resampled=True)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("jpg", "txt")
            .map(lambda x: (transform(x[0]), tokenize(x[1], truncate=True).squeeze()))
            .with_epoch(3_300_000)
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        # if is_train:
        #     loader = loader.unbatched().batched(self.batch_size)
        # else:
        #     loader = loader.unbatched().shuffle(1000).batched(self.batch_size // 2)
        return loader

    def train_dataloader(self):
        return self.make_loader(is_train=True)

    def val_dataloader(self):
        return self.make_loader(is_train=False)
