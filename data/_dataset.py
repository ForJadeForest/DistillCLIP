# 此处写对应的dataset类

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from clip import tokenize
import json
import os
import os.path as op

from PIL import Image
from clip import tokenize
from torch.utils.data import Dataset
from torchvision import transforms


class TextDataset(Dataset):
    def __init__(self, cache_dir='cache', data_dir=r'data/ref', train=True, overwrite=False,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        super(TextDataset, self).__init__()
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

    def encode_images(self, path_list):
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
            image_rep = self.encode_images(path_list)
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
                 img_std=(0.229, 0.224, 0.225)):
        super(ImageDataset, self).__init__()
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        self.path_list = None
        if not train:
            self.tokenizer = tokenize
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value.

        if self.train:
            train_image_file_list_path = op.join(self.data_dir, 'COCO', 'train2017')
            self.path_list = [op.join(train_image_file_list_path, i) for i in os.listdir(train_image_file_list_path)]
        else:
            val_image_file_list_path = op.join(self.data_dir, 'COCO', 'val2017')
            self.path_list = []
            self.captions = []
            self.sentence = []
            self.annotations_dir = op.join(self.data_dir, 'COCO', 'annotations')
            with open(op.join(self.annotations_dir, 'captions_val2017.json'), 'r') as f:
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
                    self.path_list.append(op.join(val_image_file_list_path, file_name))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]

        img = Image.open(path).convert('RGB')
        # try:
        #     img = jpeg4py.JPEG(path).decode()
        #     img = transforms.ToPILImage()(img)
        # except:
        #     img = Image.open(path).convert('RGB')
        # img = Image.fromarray(img)
        trans = transforms.Compose([

            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
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
