import json
import os.path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .rand_augment import RandAugment
from .utils import IMAGE_MEAN, IMAGE_STD


class CC3mDataset(Dataset):
    def __init__(self, image_folder, caption_json_path, train=True):
        self.img_mean, self.img_std = IMAGE_MEAN, IMAGE_STD
        self.trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            RandAugment(num_ops=4),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ])
        self.image_folder = image_folder
        self.caption_json_path = caption_json_path
        with open(caption_json_path, 'r')as f:
            caption_data = json.load(f)

        self.id2filename = {}
        self.id2caption = {}

        for single_data in caption_data:
            img_id = single_data['img_id']
            self.id2caption[img_id] = single_data['caption'][0]
            self.id2filename[img_id] = single_data['image'].split('/')[-1]
        self.id_list = self.id2filename.keys()
        self.image_path_list = [os.path.join(image_folder, self.id2filename[i]) for i in self.id_list]

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, item):
        idx = self.id_list[item]
        image_path = self.image_path_list[idx]
        text = self.id2caption[idx]
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.trans(img)
        return img_tensor, text
