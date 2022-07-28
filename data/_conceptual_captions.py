import json
from collections import OrderedDict

import torch
from PIL import Image
from pathlib2 import Path
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm, trange
from datasets import load_dataset
from io import StringIO
from PIL import Image
import requests
import urllib


from urllib.request import urlopen
"""
1. 只取出文字
    需要进行tokenize
2. 只取出图片
    需要进行transformer
3. 二者都要
"""


class ConceptualCaptions(Dataset):
    def __init__(self, data_dir, cache_dir=None, is_train=True, data_type='all', overwrite=False):
        """

        :param data_dir: The path to COCO2017
        :param is_train:
        :param data_type:
        """
        self.data = OrderedDict({
            'tokenized': [],
            'captions': [],
            'image_path_list': []
        })

        self.para = {
            'aug_prob': 0.5,
            'img_mean': (0.485, 0.456, 0.406),
            'img_std': (0.229, 0.224, 0.225)
        }
        self.overwrite = overwrite
        self.data_type = data_type
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        if is_train:
            self.mode = 'train'
        else:
            self.mode = 'validation'
        self.init_cache()
        self.load_data()
        self.text_process()

    def load_data(self):
        cache_file = self.cache_dir / 'CC_image_captions_{}.pth'.format(self.mode)
        print('load the exist data {}'.format(cache_file))
        if self.data_type == 'image':
            self.data['image_path_list'] = torch.load(str(cache_file))[1]
        elif self.data_type == 'text':
            self.data['captions'] = torch.load(str(cache_file))[0]
        elif self.data_type == 'all':
            self.data['captions'], self.data['image_path_list'] = torch.load(str(cache_file))

    def __len__(self):
        if self.data_type == 'image' or self.data_type == 'all':
            return len(self.data['image_path_list'])
        elif self.data_type == 'text':
            return len(self.data['captions'])

    def __getitem__(self, idx):
        if self.data_type == 'image':
            image_tensor = self.image_process(self.data['image_path_list'][idx])
            return image_tensor
        elif self.data_type == 'text':
            tokenized = self.data['tokenized'][idx]
            return tokenized
        elif self.data_type == 'all':
            image_tensor = self.image_process(self.data['image_path_list'][idx])
            tokenized = self.data['tokenized'][idx]
            captions = self.data['captions'][idx]
            return image_tensor, tokenized, captions

    def text_process(self):
        cache_file = self.cache_dir / 'COCO2017_tokenized_{}2017.pth'.format(self.mode)
        if cache_file.exists() and not self.overwrite:
            print('load the exist data {}'.format(cache_file))
            self.data['tokenized'] = torch.load(str(cache_file))
            return

        from clip import tokenize
        if not self.data['captions']:
            return
        print('begin tokenize.....')
        for caption in tqdm(self.data['captions']):
            self.data['tokenized'].append(tokenize(caption).squeeze())
        torch.save(self.data['tokenized'], str(cache_file))

    def load_image(self):
        train_image_file_paths: Path = self.data_dir / '{}2017'.format(self.mode)
        path_list = [f for f in train_image_file_paths.iterdir()]
        return path_list

    def load_captions(self):

        captions = []
        coco2017_file = self.data_dir / 'annotations' / 'captions_{}2017.json'.format(self.mode)

        with coco2017_file.open('r', encoding='utf8') as f:
            res = json.load(f)
        for annotation in res['annotations']:
            captions.append(annotation['caption'])
        for caption in captions:
            captions.append(caption)
        cache_file = self.cache_dir / 'captions_{}2017.pth'.format(self.mode)
        torch.save(captions, str(cache_file))
        return captions

    def init_cache(self):
        data = load_dataset('datasets/conceptual captions/conceptual_captions.py', split=self.mode)
        self.data['image_path_list'] = data['image_url']
        self.data['captions'] = data['caption']

    def image_process(self, url):
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(self.para['aug_prob']),
            transforms.RandomVerticalFlip(self.para['aug_prob']),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(self.para['img_mean'], self.para['img_std']),
        ]) if self.mode == 'train' else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.para['img_mean'], self.para['img_std'])
        ])
        response = requests.get(url)
        # img = Image.open(urlopen(url)).convert('RGB')
        img = Image.open(StringIO(urllib.requests.urlopen(url).read()))
        Image.open(urllib2.urlopen(url))
        Image.open(requests.get(url, stream=True).raw)
        img_tensor = trans(img)
        return img_tensor


if __name__ == '__main__':
    dataset = ConceptualCaptions(r'/data/pyz/data/CC', cache_dir='./', is_train=True, data_type='all', overwrite=False)
    for i in trange(len(dataset)):
        a = dataset[i]
