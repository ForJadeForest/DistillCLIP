import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import encode_images

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class TextDataset(Dataset):
    def __init__(self, cache_dir='cache', data_dir=r'data/ref', train=True, overwrite=False, teacher_name='ViT-B/32',
                 img_mean=(0.48145466, 0.4578275, 0.40821073),
                 img_std=(0.26862954, 0.26130258, 0.27577711)
                 ):
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
            self.teacher_name = teacher_name
            self.img_mean = img_mean
            self.img_std = img_std
            self.sentences, self.captions, self.path_list, self.image_rep = self.load(overwrite)
            self.image_rep = self.image_rep.squeeze(dim=1)

    def process(self):
        if self.train:
            raw_text = []
            coco2017_file = self.data_dir / 'COCO' / 'annotations' / 'captions_train2017.json'
            cc_file = self.data_dir / 'CC' / 'Train_GCC-training.tsv'
            logging.info(f'read coco2017 text data: {str(coco2017_file)}')
            with cc_file.open('r', encoding='utf8') as f:
                for content in f.readlines():
                    raw_text.append(content.split('\t')[0])
            with coco2017_file.open('r', encoding='utf8') as f:
                res = json.load(f)
                for annotation in res['annotations']:
                    raw_text.append(annotation['caption'])

            logging.info('All data: {} Begin tokenizing...'.format(len(raw_text)))
            tokenize_text = []
            for text in tqdm(raw_text):
                tokenize_text.append(self.tokenizer(text, truncate=True).squeeze())
            return torch.stack(tokenize_text)
        else:
            val_image_file_list_path = self.data_dir / 'COCO' / 'val2017'
            path_list = []
            tokenized_sentence = []
            captions = []
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
                    captions.append(caption)
                    tokenized_sentence.append(self.tokenizer(caption, truncate=True).squeeze())
                    path_list.append(val_image_file_list_path / file_name)
            image_rep = encode_images(path_list, self.teacher_name)
            return captions, tokenized_sentence, path_list, image_rep

    def load(self, overwirite):
        cache_path = self.cache_dir / f'cache-train-{self.teacher_name}.pth' \
            if self.train else self.cache_dir / f'cache-val-{self.teacher_name}.pth'

        if overwirite or not cache_path.exists():
            logging.info('重写/不存在缓存文件，开始处理文件')
            if self.train:
                tokenize_text = self.process()
                torch.save({'data_set': tokenize_text}, cache_path)
                return tokenize_text
            else:
                captions, tokenized_sentence, path_list, image_rep = self.process()
                torch.save({
                    'data_set': [
                        captions,
                        tokenized_sentence,
                        path_list,
                        image_rep
                    ]
                }, cache_path)
                return captions, tokenized_sentence, path_list, image_rep
        else:
            logging.info('直接加载缓存文件')
            data = torch.load(cache_path)['data_set']
            logging.info('加载完成！')
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
