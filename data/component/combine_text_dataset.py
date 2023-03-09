import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import encode_images, IMAGE_STD, IMAGE_MEAN

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def prepare(prepare_args):
    """
    generate the cache file for text encoder
    :param prepare_args: a Dict.
        Need contain raw_data_dir: the mscoco2017 folder path
                     cache_dir: the cache dir
                     teacher_name: the teacher name of clip model
                     overwrite: whether to overwrite the cache file
                     text_use: the text data you want to use
    :return:
    """
    from clip import tokenize
    cache_dir = Path(prepare_args['cache_dir'])
    raw_data_dir = Path(prepare_args['raw_data_dir'])
    teacher_name = prepare_args['teacher_name']
    overwrite = prepare_args['overwrite']
    text_use = prepare_args['text_use']

    train_cache_path = cache_dir / f'text-cache-train-{teacher_name.replace("/", "-")}.pth'
    val_cache_path = cache_dir / f'text-cache-val-{teacher_name.replace("/", "-")}.pth'
    if overwrite or not train_cache_path.exists():
        logging.info('overwrite/no exist the Train data cache file, begin process...')
        raw_text = []
        coco2017_file = raw_data_dir / 'mscoco' / 'annotations' / 'captions_train2017.json'
        cc_file = raw_data_dir / 'cc' / 'train_cc3m.tsv'
        if 'cc' in text_use:
            logging.info(f'read coco2017 text data: {str(coco2017_file)}')
            with cc_file.open('r', encoding='utf8') as f:
                for content in f.readlines():
                    raw_text.append(content.split('\t')[0])
        if 'coco' in text_use:
            with coco2017_file.open('r', encoding='utf8') as f:
                res = json.load(f)
                for annotation in res['annotations']:
                    raw_text.append(annotation['caption'])

        logging.info('All data: {} Begin tokenizing...'.format(len(raw_text)))
        tokenize_text = []
        for text in tqdm(raw_text):
            tokenize_text.append(tokenize(text, truncate=True).squeeze())

        tokenize_text = torch.stack(tokenize_text)
        torch.save(tokenize_text, train_cache_path)

    if overwrite or not val_cache_path.exists():
        logging.info('overwrite/no exist the Val data cache file, begin process...')
        val_image_file_list_path = raw_data_dir / 'mscoco' / 'val2017'
        path_list = []
        tokenized_sentence = []
        captions = []
        file_dir = raw_data_dir / 'mscoco' / 'annotations' / 'captions_val2017.json'
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
                tokenized_sentence.append(tokenize(caption, truncate=True).squeeze())
                path_list.append(val_image_file_list_path / file_name)
        image_rep = encode_images(path_list, teacher_name)
        torch.save([captions, tokenized_sentence, path_list, image_rep], val_cache_path)
    logging.info('Cache generation done!')


class CombineTextDataset(Dataset):
    def __init__(self, cache_dir='./.cache', train=True, teacher_name='ViT-B/32'):
        """

        :param cache_dir: the cache file dir
        :param train:
        :param teacher_name: The teacher name of CLIP
        """
        super(CombineTextDataset, self).__init__()
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        self.train = train
        self.teacher_name = teacher_name

        cache_path = self.cache_dir / f'text-cache-train-{self.teacher_name.replace("/", "-")}.pth' \
            if self.train else self.cache_dir / f'text-cache-val-{self.teacher_name.replace("/", "-")}.pth'
        logging.info('load the cache')
        if self.train:
            self.tokenize_text = torch.load(cache_path)
        else:
            self.img_mean = IMAGE_MEAN
            self.img_std = IMAGE_STD
            self.sentences, self.captions, self.path_list, self.image_rep = torch.load(cache_path)
            self.image_rep = self.image_rep.squeeze(dim=1)
        logging.info('load the cache done！')

    def __len__(self):
        if self.train:
            return len(self.tokenize_text)
        else:
            return len(self.path_list)

    def __getitem__(self, idx):
        if self.train:
            return self.tokenize_text[idx]

        return self.image_rep[idx], self.captions[idx], self.sentences[idx]
