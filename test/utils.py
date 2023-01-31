import torch
from model.component.clip_model import CLIPModel
from model._utils import teacher_load


def load_image_encoder(cpk_path):
    return torch.load(cpk_path)


def load_text_encoder(cpk_path):
    return torch.load(cpk_path)


def get_model(image_path=None, text_path=None) -> CLIPModel:
    if image_path is None and text_path is None:
        image_encoder = load_image_encoder(image_path)
        text_encoder = load_text_encoder(text_path)
        clip_model = CLIPModel(False, image_encoder, text_encoder)
    else:
        print("[INFO] ==> Now load the original clip model!")
        clip_model = teacher_load('Vit-B/32', '/data/pyz/.cache', model_type='all')
    return clip_model
