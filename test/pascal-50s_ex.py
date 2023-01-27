from random import random

import torch
import scipy
import os

from PIL.Image import Image
from torchvision.transforms import transforms


class Pascal50sDataset(torch.utils.data.Dataset):
    idx2cat = {1: 'HC', 2: 'HI', 3: 'HM', 4: 'MM'}

    def __init__(self,
                 root: str = "data/Pascal-50s/",
                 media_size: int = 224,
                 voc_path: str = "data/VOC2010/"):
        super().__init__()
        self.voc_path = voc_path
        self.read_data(root)
        self.read_score(root)
        # self.transforms = keys_to_transforms([], size=media_size)
        self.transforms = transforms.Compose([
            transforms.Resize(media_size, media_size)
        ])

    @staticmethod
    def loadmat(path):
        return scipy.io.loadmat(path)

    def read_data(self, root):
        mat = self.loadmat(
            os.path.join(root, "pyCIDErConsensus/pair_pascal.mat"))
        self.data = mat["new_input"][0]
        self.categories = mat["category"][0]
        # sanity check
        c = torch.Tensor(mat["new_data"])
        hc = (c.sum(dim=-1) == 12).int()
        hi = (c.sum(dim=-1) == 13).int()
        hm = ((c < 6).sum(dim=-1) == 1).int()
        mm = ((c < 6).sum(dim=-1) == 2).int()
        assert 1000 == hc.sum()
        assert 1000 == hi.sum()
        assert 1000 == hm.sum()
        assert 1000 == mm.sum()
        assert (hc + hi + hm + mm).sum() == self.categories.shape[0]
        chk = (torch.Tensor(self.categories) - hc - hi * 2 - hm * 3 - mm * 4)
        assert 0 == chk.abs().sum(), chk

    def read_score(self, root):
        mat = self.loadmat(
            os.path.join(root, "pyCIDErConsensus/consensus_pascal.mat"))
        data = mat["triplets"][0]
        self.labels = []
        self.references = []
        for i in range(len(self)):
            votes = {}
            refs = []
            for j in range(i * 48, (i + 1) * 48):
                a,b,c,d = [x[0][0] for x in data[j]]
                key = b[0].strip() if 1 == d else c[0].strip()
                refs.append(a[0].strip())
                votes[key] = votes.get(key, 0) + 1
            assert 2 >= len(votes.keys()), votes
            assert len(votes.keys()) > 0
            try:
                vote_a = votes.get(self.data[i][1][0].strip(), 0)
                vote_b = votes.get(self.data[i][2][0].strip(), 0)
            except KeyError:
                print("warning: data mismatch!")
                print(f"a: {self.data[i][1][0].strip()}")
                print(f"b: {self.data[i][2][0].strip()}")
                print(votes)
                exit()
            # Ties are broken randomly.
            label = 0 if vote_a > vote_b + random.random() - .5 else 1
            self.labels.append(label)
            self.references.append(refs)

    def __len__(self):
        return len(self.data)

    def get_image(self, filename: str):
        path = os.path.join(self.voc_path, "JPEGImages")
        img = Image.open(os.path.join(path, filename)).convert('RGB')
        return self.transforms(img)

    def __getitem__(self, idx: int):
        vid, a, b = [x[0] for x in self.data[idx]]
        label = self.labels[idx]
        feat = self.get_image(vid)
        a = a.strip()
        b = b.strip()
        references = self.references[idx]
        category = self.categories[idx]
        return feat, a, b, references, category, label