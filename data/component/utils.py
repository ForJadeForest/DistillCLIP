import torch
from PIL import Image
from tqdm import tqdm

IMAGE_DATASET_NAME = ['coco', 'data_256', 'imagenet']
IMAGE_PREFIX = {
    'coco': '0',
    'data_256': 'data_256',
    'imagenet': 'imagenet'
}


def encode_images(path_list, teacher_name: str):
    from clip import load
    image_encode = []
    device = 'cuda'
    model, preprocess = load(teacher_name, device)
    model.eval()
    for path in tqdm(path_list):
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).float()
            image_encode.append(image_features)
    return torch.cat(image_encode, dim=0)


def encode_texts(caption_list, teacher_name: str):
    from clip import load, tokenize
    text_encode = []
    device = 'cuda'
    model, preprocess = load(teacher_name, device)
    model.eval()
    for caption in tqdm(caption_list):
        with torch.no_grad():
            caption = tokenize(caption).to(device)
            image_features = model.encode_text(caption).float().to('cpu')
            text_encode.append(image_features)
    return torch.cat(text_encode, dim=0)
