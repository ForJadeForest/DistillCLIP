import json
import os.path as op
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def encode_images(path_list):
    from clip import load
    image_encode = []
    for path in tqdm(path_list):
        device = 'cuda'
        model, preprocess = load("ViT-B/32", device=device)
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image.to(device)).float()
            image_encode.append(image_features)
    return torch.cat(image_encode, dim=0)


class TextDataset(Dataset):
    def __init__(self, cache_dir='cache', data_dir=r'data/ref', train=True, overwrite=False,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        super(TextDataset, self).__init__()
        from clip import tokenize
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        self.train = train
        self.tokenizer = tokenize
        if self.train:
            self.tokenize_text = self.load(overwrite)
        else:
            self.img_mean = img_mean
            self.img_std = img_std
            self.sentences, self.captions, self.path_list, self.image_rep = self.load(overwrite)
            self.image_rep = self.image_rep.squeeze(dim=1)

    def process(self):
        raw_text = []
        if self.train:
            coco2017_file = self.data_dir / 'COCO' / 'annotations' / 'captions_train2017.json'
            cc_file = self.data_dir / 'CC' / 'Train_GCC-training.tsv'
            print(coco2017_file)
            with cc_file.open('r', encoding='utf8') as f:
                for content in f.readlines():
                    raw_text.append(content.split('\t')[0])
            with coco2017_file.open('r', encoding='utf8') as f:
                res = json.load(f)
                for annotation in res['annotations']:
                    raw_text.append(annotation['caption'])

            print('All data: {} Begin tokenizing...'.format(len(raw_text)))
            tokenize_text = []
            for text in tqdm(raw_text):
                try:
                    tokenize_text.append(self.tokenizer(text).squeeze())
                except:
                    continue
            print('Tokenize Done! with {} texts loads'.format(len(tokenize_text)))
            print('the rate is {}'.format(len(tokenize_text) / len(raw_text)))

            return torch.stack(tokenize_text)
        else:
            val_image_file_list_path = self.data_dir / 'COCO' / 'val2017'
            path_list = []
            captions = []
            sentences = []
            file_dir = self.data_dir / 'COCO' / 'annotations' / 'captions_val2017.json'
            with file_dir.open('r', encoding='utf8') as f:
                data = json.load(f)
            images = data['images']
            id2caption = {}
            id2filename = {}
            for image in images:
                id2filename[image['id']] = image['file_name']
            for annotation in data['annotations']:
                id2caption[annotation['image_id']] = annotation['caption']
            for id, file_name in id2filename.items():
                caption = id2caption.get(id, None)
                if caption:
                    sentences.append(caption)
                    captions.append(self.tokenizer(caption).squeeze())
                    path_list.append(val_image_file_list_path / file_name)
            image_rep = encode_images(path_list)
            return sentences, captions, path_list, image_rep

    def load(self, overwirite):
        cache_path = self.cache_dir / 'cache-train.pth' if self.train else self.cache_dir / 'cache-val.pth'
        if overwirite or not cache_path.exists():
            print('重写/不存在缓存文件，开始处理文件')
            if self.train:
                tokenize_text = self.process()
                torch.save({'data_set': tokenize_text}, cache_path)
                return tokenize_text
            else:
                sentences, captions, path_list, image_rep = self.process()
                torch.save({
                    'data_set': [
                        sentences,
                        captions,
                        path_list,
                        image_rep
                    ]
                }, cache_path)
                return sentences, captions, path_list, image_rep
        else:
            print('直接加载缓存文件')
            data = torch.load(cache_path)['data_set']
            print('加载完成！')
            return data

    def __len__(self):
        if self.train:
            return len(self.tokenize_text)
        else:
            return len(self.path_list)

    def __getitem__(self, idx):
        if self.train:
            return self.tokenize_text[idx]

        return self.image_rep[idx], self.captions[idx], self.sentences[idx]


class ImageDataset(Dataset):
    def __init__(self, data_dir=r'data/ref', train=True, no_augment=True, aug_prob=0.5, img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225), train_dir=None, need_crop=False, image_use='all'):
        super(ImageDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.train = train
        self.no_augment = no_augment
        self.aug_prob = aug_prob
        self.img_mean = img_mean
        self.img_std = img_std
        self.train_dir = Path(train_dir)
        self.aug = train and not no_augment
        self.path_list = None
        self.need_crop = need_crop

        if not train:
            from clip import tokenize
            self.tokenizer = tokenize
            self.val_image_file_list_path = self.data_dir / 'COCO' / 'val2017'
            self.load_validation_data()
        else:
            if self.train_dir:
                self.train_image_file_path = self.train_dir
            else:
                self.train_image_file_path = self.data_dir / 'COCO' / 'train2017'
            if image_use == 'all':
                self.path_list = [path for path in self.train_dir.iterdir()]
            else:
                self.path_list = [path for path in self.train_dir.iterdir() if
                                not str(path.name).startswith('data_256')]

    def load_validation_data(self):
        self.path_list = []
        self.captions = []
        self.sentence = []
        self.annotations_dir = self.data_dir / 'COCO' / 'annotations'

        with open((self.annotations_dir / 'captions_val2017.json'), 'r') as f:
            data = json.load(f)
        images = data['images']
        id2caption = {}
        id2filename = {}
        for image in images:
            id2filename[image['id']] = image['file_name']
        for annotation in data['annotations']:
            id2caption[annotation['image_id']] = annotation['caption']
        for id, file_name in id2filename.items():
            caption = id2caption.get(id, None)
            if caption:
                self.sentence.append(caption)
                self.captions.append(self.tokenizer(caption).squeeze())
                self.path_list.append(op.join(self.val_image_file_list_path / file_name))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')
        if self.need_crop:
            crop_trans = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ])
            img = crop_trans(img)

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
            return img_tensor, self.captions[idx], self.sentence[idx]


class ImageDatasetLmdb(Dataset):
    def __init__(self, data_dir=r'data/ref', train=True, no_augment=True, aug_prob=0.5, img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225), train_dir=None, need_crop=False, lmdb_root='/data/pyz/ECSSD.lmdb'):
        super(ImageDatasetLmdb, self).__init__()
        self.data_dir = Path(data_dir)
        self.train = train
        self.no_augment = no_augment
        self.aug_prob = aug_prob
        self.img_mean = img_mean
        self.img_std = img_std
        self.train_dir = Path(train_dir)
        self.aug = train and not no_augment
        self.path_list = None
        self.lmdb_root = lmdb_root
        self.need_crop = need_crop

        if not train:
            from clip import tokenize
            self.tokenizer = tokenize
            self.val_image_file_list_path = self.data_dir / 'COCO' / 'val2017'
            self.load_validation_data()
        else:
            if self.train_dir:
                self.train_image_file_path = self.train_dir

            else:
                self.train_image_file_path = self.data_dir / 'COCO' / 'train2017'
            self.path_list = list(self.train_image_file_path.iterdir())
            self.imgs = []
            self.img_env = lmdb.open(self.lmdb_root, readonly=True, lock=False, readahead=False,
                                     meminit=False)
            for file_name in tqdm(self.train_image_file_path.iterdir(),
                                  total=len(list(self.train_image_file_path.iterdir()))):
                self.imgs.append(_read_img_lmdb(self.img_env, file_name.name, (224, 224, 3)))

    def load_validation_data(self):
        self.path_list = []
        self.captions = []
        self.sentence = []
        self.imgs = []
        self.annotations_dir = self.data_dir / 'COCO' / 'annotations'

        with open((self.annotations_dir / 'captions_val2017.json'), 'r') as f:
            data = json.load(f)
        images = data['images']
        id2caption = {}
        id2filename = {}
        for image in images:
            id2filename[image['id']] = image['file_name']
        for annotation in data['annotations']:
            id2caption[annotation['image_id']] = annotation['caption']
        for id, file_name in id2filename.items():
            caption = id2caption.get(id, None)
            if caption:
                self.sentence.append(caption)
                self.captions.append(self.tokenizer(caption).squeeze())
                self.path_list.append(op.join(self.val_image_file_list_path / file_name))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        if self.train:
            file_name = Path(path).name
            # img_arr = _read_img_lmdb(self.img_env, file_name, (224, 224, 3))
            img_arr = self.imgs
        else:
            img_arr = Image.open(path).convert('RGB')

        if self.need_crop:
            crop_trans = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ])
            img_arr = crop_trans(img_arr)

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ]) if self.train else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])

        img_tensor = trans(img_arr)

        if self.train:
            return img_tensor
        else:
            return img_tensor, self.captions[idx], self.sentence[idx]


def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    arr = np.frombuffer(buf, dtype=np.uint8)  # arr shape: [channel * height * weight, ]
    img_arr = arr.reshape(size)  # should be height * weight * channel
    img_arr = img_arr[:, :, [2, 1, 0]]  # bgr => rgb
    return img_arr


if __name__ == '__main__':
    from clip import tokenize
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    text_data = TextDataset('/home/pyz/data/cache', '/home/pyz/data', True, tokenize, True)
    for data in DataLoader(text_data, batch_size=2):
        print(data)
        break
    text_data = TextDataset('/home/pyz/data/cache', '/home/pyz/data', False, tokenize, False)
    for data in DataLoader(text_data, batch_size=1, shuffle=True):
        image, caption, sentence = data
        plt.imshow(image.squeeze(0).permute(1, 2, 0))
        print(caption, sentence)
        break
