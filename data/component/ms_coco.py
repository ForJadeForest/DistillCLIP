import os.path

from torchvision import datasets, transforms
from data.component.rand_augment import RandAugment
from data.component.utils import IMAGE_MEAN, IMAGE_STD


class COCODataset(datasets.CocoCaptions):
    def __init__(self, root_path, annotation_path, need_type='all', need_text_processor=True, train=True, version=2014):
        from clip import tokenize
        self.need_type = need_type
        self.train = train
        self.tokenizer = tokenize
        self.img_mean, self.img_std = IMAGE_MEAN, IMAGE_STD
        self.need_text_processor = need_text_processor
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
        if train:
            root = os.path.join(root_path, f'train{version}')
            annotation_file = os.path.join(annotation_path, f'captions_train{version}.json')
        else:
            root = os.path.join(root_path, f'val{version}')
            annotation_file = os.path.join(annotation_path, f'captions_val{version}.json')
        super(COCODataset, self).__init__(root, annotation_file, self.trans)

    def __getitem__(self, item):
        image, caption = super(COCODataset, self).__getitem__(item)
        if self.need_text_processor:
            caption = self.tokenizer(caption[0])[0]
        else:
            caption = caption[0]
        if self.need_type == 'all' or not self.train:
            return image, caption
        elif self.need_type == 'image':
            return image
        elif self.need_type == 'text':
            return caption
        else:
            raise ValueError('the mscoco dataset need_type parameter should is [\'all\', \'text\', \'image\'], '
                             f'bug get {self.need_type}')
