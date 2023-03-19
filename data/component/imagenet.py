import os.path

from timm.data.parsers.parser_image_in_tar import ParserImageInTar
from timm.data import ImageDataset
from torchvision import transforms
from .rand_augment import RandAugment
from .utils import IMAGE_MEAN, IMAGE_STD


class ImageNetDataset(ImageDataset):
    def __init__(self, data_path, train=True, need_label=False, use_transform=True):
        self.img_mean, self.img_std = IMAGE_MEAN, IMAGE_STD
        self.need_label = need_label
        self.trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            RandAugment(num_ops=4),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ]) if train else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        if not use_transform:
            self.trans = None
        if train:
            data_path = os.path.join(data_path, 'train')
            super(ImageNetDataset, self).__init__(data_path, ParserImageInTar(data_path), transform=self.trans)
        else:
            data_path = os.path.join(data_path, 'val')
            super(ImageNetDataset, self).__init__(data_path, transform=self.trans)

    def __getitem__(self, item):
        image, label = super(ImageNetDataset, self).__getitem__(item)
        if self.need_label:
            return image, label
        return image
