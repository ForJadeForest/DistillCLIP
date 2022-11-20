import os.path

from torchvision import datasets, transforms

from .utils import IMAGE_MEAN, IMAGE_STD


class COCODataset(datasets.CocoCaptions):
    def __init__(self, root_path, annotation_path, train=True):
        from clip import tokenize
        self.tokenizer = tokenize
        self.img_mean, self.img_std = IMAGE_MEAN, IMAGE_STD
        self.trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandAugment(num_ops=4),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ]) if train else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        if train:
            root = os.path.join(root_path, 'train2017')
            annotation_file = os.path.join(annotation_path, 'captions_train2017.json')
        else:
            root = os.path.join(root_path, 'val2017')
            annotation_file = os.path.join(annotation_path, 'captions_val2017.json')
        super(COCODataset, self).__init__(root, annotation_file, self.trans)

    def __getitem__(self, item):
        # TODO: can make use the other captions for one image
        image, caption = super(COCODataset, self).__getitem__(item)
        image, caption = image, self.tokenizer(caption[0], truncate=False)[0]
        return image, caption
