import os
from random import random

import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Pascal50sDataset(Dataset):

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
            transforms.Resize(media_size)
        ])

    def read_data(self, root):
        mat = loadmat(
            os.path.join(root, "pyCIDErConsensus/pair_pascal.mat"))
        # self.data 中每一行包含了 image_id, sent1, sent2，一共有4000个句子对
        # self.categories 中包含句子对的标签？有四种 {1: 'HC', 2: 'HI', 3: 'HM', 4: 'MM'}
        # HC 是两个人类正确句子对
        # HI 是两个人类错误的句子对（意思是一个句子描述图像，另一个句子描述另外一张图像）
        # HM 是一个人类描述的句子和一个机器生成描述同一张图像的句子
        # MM 是描述两个机器生成的描述同一张图像的句子
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
        mat = loadmat(
            os.path.join(root, "pyCIDErConsensus/consensus_pascal.mat"))
        # pascal-50s 就是每一张图像有50个ref
        # data 192000（4000 * 48）个数据，是人工标注的ref
        # 每一行有三句话 + 一个 rate 列
        # 其中第一句话应该是ref，第二三句是描述该图像的50ref除去这48个之后得到的，也就是每48个数据B，C列的都是相同的（就是那4k个句子对）
        # 对每一个ref，测评员需要去对其进行选择B，C哪一个和A更加相似。
        # 因此对于一张image的48个标准 ref 进行投票，最终票数多的作为label（个人估计实验中就是把他作为candidate）
        data = mat["triplets"][0]
        self.labels = []
        self.references = []
        for i in range(len(self)):
            votes = {}
            refs = []
            for j in range(i * 48, (i + 1) * 48):
                a, b, c, d = [x[0][0] for x in data[j]]
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
            label = 0 if vote_a > vote_b else 1
            self.labels.append(label)
            self.references.append(refs)

    def __len__(self):
        return len(self.data)

    def get_image(self, filename: str):
        path = os.path.join(self.voc_path, "JPEGImages")
        path = os.path.join(path, filename)
        # img = Image.open(os.path.join(path, filename)).convert('RGB')
        # return self.transforms(img)
        return path

    def __getitem__(self, idx: int):
        vid, a, b = [x[0] for x in self.data[idx]]
        label = self.labels[idx]
        feat = self.get_image(vid)
        # feat = 0
        a = a.strip()
        b = b.strip()
        references = self.references[idx]
        category = self.categories[idx]
        return feat, a, b, references, category, label
