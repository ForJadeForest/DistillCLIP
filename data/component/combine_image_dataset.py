import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .rand_augment import RandAugment
from .utils import IMAGE_DATASET_NAME, IMAGE_PREFIX, IMAGE_MEAN, IMAGE_STD, encode_texts

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def prepare(prepare_args):
    """
    用来下载数据，或者生成缓存数据
    :param prepare_args:
    :return:
    """
    raw_data_dir = Path(prepare_args['raw_data_dir'])
    cache_dir = Path(prepare_args['cache_dir'])
    teacher_name = prepare_args['teacher_name']
    overwrite = prepare_args['overwrite']

    cache_path = cache_dir / f'cache-val-{teacher_name.replace("/", "-")}.pth'
    if not cache_path.exists() or overwrite:
        logging.info('the cache_dir not exists or you set overwrite')
        val_image_file_list_path = raw_data_dir / 'mscoco' / 'val2017'
        path_list = []
        captions = []
        annotations_dir = raw_data_dir / 'mscoco' / 'annotations'
        with open((annotations_dir / 'captions_val2017.json'), 'r') as f:
            coco_data = json.load(f)
        images = coco_data['images']
        id2caption = {}
        id2filename = {}
        for image in images:
            id2filename[image['id']] = image['file_name']
        for annotation in coco_data['annotations']:
            id2caption[annotation['image_id']] = annotation['caption']
        for id, file_name in id2filename.items():
            caption = id2caption.get(id, None)
            if caption:
                captions.append(caption)
                path_list.append(val_image_file_list_path / file_name)
        captions_rep = encode_texts(captions, teacher_name)
        torch.save([path_list, captions_rep, captions], cache_path)
        logging.info(f'cache data saved in {str(cache_path)}')


class CombineImageDataset(Dataset):
    def __init__(self, combine_dataset_path, train=True, image_use=None, cache_dir='cache', teacher_name='ViT-B/32'):
        super(CombineImageDataset, self).__init__()
        if image_use is None:
            image_use = ['coco', 'data_256', 'imagenet']

        for i in image_use:
            assert i in IMAGE_DATASET_NAME, f'the {i} dataset name is not exists in {IMAGE_DATASET_NAME}'
        self.train = train
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        self.cache_dir = Path(cache_dir)
        self.teacher_name = teacher_name

        if not train:
            cache_path = self.cache_dir / f'cache-val-{self.teacher_name.replace("/", "-")}.pth'
            self.path_list, self.captions_rep, self.captions = torch.load(cache_path)
        else:
            self.train_image_file_path = Path(combine_dataset_path)

            def filter_dataset(x):
                res = False
                for name in image_use:
                    prefix = IMAGE_PREFIX[name]
                    res = res or x.startswith(prefix)
                return res

            self.path_list = [path for path in self.train_image_file_path.iterdir() if filter_dataset(path.name)]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')

        trans = transforms.Compose([
            RandAugment(num_ops=4),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ]) if self.train else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])

        img_tensor = trans(img)

        if self.train:
            return img_tensor
        else:
            return img_tensor, self.captions_rep[idx], self.captions[idx]
