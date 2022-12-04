import json
import os
from pathlib import Path

from PIL import Image
from clip import tokenize
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data.component.utils import IMAGE_MEAN, IMAGE_STD


class TestImageDataset(Dataset):
    def __init__(self, file_path):
        super(TestImageDataset, self).__init__()
        self.file_path = file_path
        self.file_list = os.listdir(file_path)
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        photo_path = os.path.join(self.file_path, self.file_list[item])
        image = Image.open(photo_path).convert('RGB')
        return self.file_list[item], self.preprocess(image)


class TestDataset(Dataset):
    def __init__(self, data_dir, preprocess):
        super(TestDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenize
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        self.sentences, self.captions, self.path_list = self.process()
        self.preprocess = preprocess

    def process(self):

        val_image_file_list_path = self.data_dir / 'mscoco' / 'val2017'
        path_list = []
        captions = []
        sentences = []
        file_dir = self.data_dir / 'mscoco' / 'annotations' / 'captions_val2017.json'
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

        return sentences, captions, path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')
        img_tensor = self.preprocess(img)
        return self.path_list[idx].name, img_tensor, self.captions[idx], self.sentences[idx]
