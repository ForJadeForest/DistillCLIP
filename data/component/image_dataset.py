import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import IMAGE_DATASET_NAME, IMAGE_PREFIX, encode_texts

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class ImageDataset(Dataset):
    def __init__(self, data_dir=r'data/ref', train=True, no_augment=True,
                 aug_prob=0.5, image_use=None, cache_dir='cache', teacher_name='ViT-B/32', overwrite=False,
                 train_image_dir='/data/pyz/data/combine_dataset',
                 img_mean=(0.48145466, 0.4578275, 0.40821073),
                 img_std=(0.26862954, 0.26130258, 0.27577711)):
        super(ImageDataset, self).__init__()
        if image_use is None:
            image_use = ['coco', 'data_256', 'imagenet']

        for i in image_use:
            assert i in IMAGE_DATASET_NAME, f'the {i} dataset name is not exists in {IMAGE_DATASET_NAME}'
        self.data_dir = Path(data_dir)
        self.train = train
        self.no_augment = no_augment
        self.aug_prob = aug_prob
        self.img_mean = img_mean
        self.img_std = img_std
        self.cache_dir = Path(cache_dir)
        self.aug = train and not no_augment
        self.path_list = None
        self.teacher_name = teacher_name
        if not train:
            cache_path = self.cache_dir / f'cache-val-{self.teacher_name.replace("/", "-")}.pth'
            if not cache_path.exists() or overwrite:
                logging.info('the cache_dir not exists or you set overwrite')
                from clip import tokenize
                self.tokenizer = tokenize
                self.val_image_file_list_path = self.data_dir / 'COCO' / 'val2017'
                self.path_list = []
                self.captions = []
                self.annotations_dir = self.data_dir / 'COCO' / 'annotations'

                with open((self.annotations_dir / 'captions_val2017.json'), 'r') as f:
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
                        self.captions.append(caption)
                        self.path_list.append(self.val_image_file_list_path / file_name)
                self.captions_rep = encode_texts(self.captions, self.teacher_name)
                torch.save({
                    'data_set': [
                        self.path_list,
                        self.captions_rep,
                        self.captions
                    ]
                }, cache_path)
                logging.info(f'cache data saved in {str(cache_path)}')
            else:
                logging.info(f'cache data exists in {str(cache_path)}')
                self.path_list, self.captions_rep, self.captions = torch.load(cache_path)['data_set']
                logging.info(f'load cache data successfully')
        else:
            self.train_image_file_path = Path(train_image_dir)

            def filter_dataset(x):
                res = False
                for name in image_use:
                    prefix = IMAGE_PREFIX[name]
                    res = res or x.startswith(prefix)
                return res

            self.path_list = [path for path in self.train_image_file_path.iterdir() if filter_dataset(path.name)]

    def load_validation_data(self):
        self.path_list = []
        self.captions = []
        self.annotations_dir = self.data_dir / 'COCO' / 'annotations'

        with open((self.annotations_dir / 'captions_val2017.json'), 'r') as f:
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
                self.captions.append(caption)
                self.path_list.append(self.val_image_file_list_path / file_name)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')

        trans = transforms.Compose([
            transforms.RandAugment(num_ops=4),
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
