import os

import torch
from PIL import Image
from clip import tokenize
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.component.test_image_dataset import TestImageDataset
from model.component.weight_share_model import RepeatTextTransformer, RepeatVisionTransformer
from model.dual_distill_model import DualDistillModel

device = 'cuda:0'
checkpoint = r'D:\Code\Dis_CLIP\temp\cpk\895-val_acc0.076-loss0.12050.ckpt'

_visual = RepeatVisionTransformer(
    img_size=224, patch_size=32, in_chans=3, out_dim=512, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.0,
    qkv_bias=True, repeated_times=2, use_transform=True
)
_text = RepeatTextTransformer(
    out_dim=512, embed_dim=512, depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True, repeated_times=2, use_transform=True
)
model = DualDistillModel.load_from_checkpoint(checkpoint, torch.device('cpu'), image_student=_visual,
                                              text_student=_text)
image = model.student.image_encoder.to(device)
text = model.student.text_encoder.to(device)

encode_path = '../temp/encode_file'
file_path = r'D:\data\mscoco\val2017'

query = 'A woman sitting on a bench and a woman standing waiting for the bus.'
res_save_path = '2'


def imageQuery(query, image, text, encodes_path, file_path, device, save_path):
    image_set = TestImageDataset(file_path)
    if not os.path.exists(encodes_path):
        os.mkdir(encodes_path)
        for file_names, images in tqdm(DataLoader(image_set, batch_size=32, shuffle=False)):
            out = image(images.to(device))
            out = out.last_representation
            for file_name, feature in zip(file_names, out):
                save = feature / feature.norm(dim=0, keepdim=True)
                torch.save(save, os.path.join(encodes_path, file_name.split('.')[0] + '.pth'))

    encodes = []
    name = [s.split('.')[0] for s in os.listdir(encodes_path)]
    for file in os.listdir(encodes_path):
        res = torch.load(os.path.join(encodes_path, file))
        encodes.append(res)
    encodes = torch.stack(encodes, 0).to(device)
    query = tokenize(query)

    text_features = text(query.to(device)).last_representation.float()
    text_features = text_features / text_features.norm(dim=0, keepdim=True)
    text_probs = (100 * text_features @ encodes.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(20, dim=-1)
    figure = plt.figure(figsize=(10, 8), dpi=200)
    data_dir = file_path
    top_labels = top_labels.squeeze(dim=0)
    top_probs = top_probs.squeeze(dim=0)
    print(top_labels.shape)

    for i, index in enumerate(top_labels):
        filename = name[index]
        path = os.path.join(data_dir, filename + '.jpg')
        img = Image.open(path).convert("RGB")
        plt.subplot(5, 4, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title('prob: {}'.format(round(top_probs[i].item(), 4)))
    save_path = save_path + '.png'
    save_path = os.path.join('../temp/test_res', save_path)
    plt.savefig(save_path)


imageQuery(query, image, text, encode_path, file_path, device, save_path=res_save_path)
