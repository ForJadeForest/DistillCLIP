import pytorch_lightning as pl
import webdataset as wds
from sklearn.model_selection import train_test_split
from torchvision import transforms
from pathlib import Path
from clip import tokenize


class TextImageDataModule(pl.LightningDataModule):
    def __init__(self, image_path, batch_size=64, workers=4):
        super(TextImageDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = workers
        print("batch_size", self.batch_size, "num_workers", self.num_workers)
        self.img_mean = (0.48145466, 0.4578275, 0.40821073)
        self.img_std = (0.26862954, 0.26130258, 0.27577711)
        url = [str(i) for i in list(Path(image_path).glob('*.tar'))]
        self.train_url, self.val_url = train_test_split(url, test_size=0.1)

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
            dataset_size = 551335
            shuffle = 5000
        else:
            urls = self.val_url
            dataset_size = 64376
            shuffle = 0

        transform = self.make_transform(is_train)

        dataset = (
            wds.WebDataset(urls)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("jpg", "txt")
            .map(lambda x: (transform(x[0]), tokenize(x[1])))
            .batched(self.batch_size, partial=False)
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,

        )

        loader.length = dataset_size // self.batch_size
        if is_train:
            # ensure same number of batches in all clients
            loader = loader.ddp_equalize(dataset_size // self.batch_size)
            # print("# loader length", len(loader))

        return loader

    def train_dataloader(self):
        return self.make_loader(is_train=True)

    def val_dataloader(self):
        return self.make_loader(is_train=False)
