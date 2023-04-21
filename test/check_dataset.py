from model.component.clip_model import CLIPModel
from data.component.cc3m_train_dataset import get_cc3m_dataset
from test.utils.load_clip_model.default_model import CLIPModelLoadConfig
from test.utils.load_clip_model.get_model import load_version
from utils.load_model import load_model
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

model_config = CLIPModelLoadConfig(
    clip_path='/data/share/pyz/Dis_CLIP/final/clip/shtc5cml/checkpoints/287-val_acc0.241-loss0.05855.ckpt'
)
model = load_version(model_config, 'L-CLIP', device='cuda', use_fp16=True).model

dataset = get_cc3m_dataset(num_workers=6,
                           cc3m_shards='/data/pyz/data/cc/train_cc3m/{00000..00331}.tar',
                           need_text_processor=False,
                           batch_size=2048)
t = 0
with torch.no_grad():
    for _ in range(10):
        t = 0
        for i in dataset:
            with torch.autocast('cuda'):
                res1, res2,_ = model(i[1], i[0].to('cuda').half())
                if torch.isnan(res1).any():
                    print('res1 is nan!', i)
                if torch.isnan(res2).any():
                    print('res2 is nan !', i)
            print((t+1)*2048)
            t += 1
