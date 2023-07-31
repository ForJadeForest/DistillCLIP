from torch.utils.data import Dataset
from test.utils.ex_script.flickr8k_ex import load_data
from clip import tokenize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


class Flickr8kDataset(Dataset):
    def __init__(self, input_json_path, image_directory, prefix='A photo depicts', need_text_processor=True,
                 train=False):
        assert not train, f"the Flickr8k only use to validation! please set is_train as False"
        images, refs, candidates, human_scores = load_data(input_json_path, image_directory)
        self.images = images
        self.refs = refs
        self.candidates = candidates
        self.human_scores = human_scores
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '
        self.need_text_processor = need_text_processor
        self.image_processor = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ref = self.refs[item]
        if self.need_text_processor:
            ref = tokenize([self.prefix + r for r in ref])
            candidate = tokenize(self.prefix + self.candidates[item]).squeeze()
        else:
            ref = [self.prefix + r for r in ref]
            candidate = self.prefix + self.candidates[item]

        image = self.image_processor(Image.open(self.images[item]))

        return image, ref, candidate, self.human_scores[item]


if __name__ == '__main__':
    import os

    flickr8k_expert_file = os.path.join('/data/pyz/data/flickr8k', 'flickr8k.json')
    data = Flickr8kDataset(flickr8k_expert_file, '/data/pyz/data/flickr8k')
    from torch.utils.data import DataLoader

    dataloader = DataLoader(data, batch_size=108, num_workers=12)
    for i in dataloader:
        pass
