from pathlib import Path
from itertools import combinations
import torch
from model.utils import teacher_load
from model.component.clip_model import CLIPModel
from test.test_model.clip_model import TestCLIPModel
from encoder_model import *


def load_image_encoder(cpk_path):
    cpk = torch.load(cpk_path)
    visual_encoder = mini_vision_encoder()
    state_dict = {k.replace('student.', ''): v for k, v in cpk['state_dict'].items() if k.startswith('student')}
    visual_encoder.load_state_dict(state_dict)
    return visual_encoder


def load_text_encoder(cpk_path):
    try:
        cpk = torch.load(cpk_path)
        text_encoder = mini_text_encoder()
        state_dict = {k.replace('student.', ''): v for k, v in cpk['state_dict'].items() if k.startswith('student')}
        text_encoder.load_state_dict(state_dict)
    except:
        cpk = torch.load(cpk_path)
        text_encoder = mini_compression_text_encoder()
        state_dict = {k.replace('student.', ''): v for k, v in cpk['state_dict'].items() if k.startswith('student')}
        text_encoder.load_state_dict(state_dict)
    return text_encoder


def get_clip_model(device, use_fp16=True, clip_path=None, image_path=None, text_path=None,
                   load_teacher=False) -> CLIPModel:
    """
    image_path: the image encoder checkpoint path
    text_path: the text encoder checkpoint path
    device: the device for the model
    """
    if load_teacher:
        print("[INFO] ==> Now load the original clip model!")
        clip_model = teacher_load('ViT-B/32', r'/data/pyz/.cache', model_type='all', only_last_rep=True)
    elif image_path is not None and text_path is not None:
        image_encoder = load_image_encoder(image_path)
        text_encoder = load_text_encoder(text_path)
        clip_model = CLIPModel(False, image_encoder, text_encoder, only_last_rep=True)
    elif clip_path is not None:
        visual_encoder = mini_vision_encoder()
        state_dict = torch.load(clip_path)['state_dict']
        state_dict = {k.replace('student.', ''): v for k, v in state_dict.items() if k.startswith('student')}
        try:
            text_encoder = mini_text_encoder()
            clip_model = CLIPModel(False, visual_encoder, text_encoder, only_last_rep=True)
            clip_model.load_state_dict(state_dict)
        except:
            text_encoder = mini_compression_text_encoder()
            clip_model = CLIPModel(False, visual_encoder, text_encoder, only_last_rep=True)
            clip_model.load_state_dict(state_dict)
    else:
        raise ValueError(f'the clip path, image_path and text path are None!')

    # clip model default use half precision
    if use_fp16:
        clip_model.half()
    clip_model.to(device)
    clip_model.eval()
    clip_model.only_last_rep = True

    return clip_model


def get_all_version_path(root_path, model_name):
    """
    获得一个模型的所有version的路径
    :param root_path:
    :param model_name:
    :return:
    """
    # root_path = '/data/share/pyz/Dis_CLIP/final'
    root_path = Path(root_path)
    model_name_path_map = {
        'WP&Single': {
            'image_path': root_path / 'image' / 'wp_single',
            'text_path': root_path / 'text' / 'wp_single'
        },
        'WP&SR': {
            'clip_path': root_path / 'clip' / 'wp_sr'
        },
        'WP+MD&Single': {
            'image_path': root_path / 'image' / 'wp_single',
            'text_path': root_path / 'text' / 'wp_md_single'
        },
        'L-CLIP': {
            'clip_path': root_path / 'clip' / 'l_clip'
        },
        'CE-CLIP': {
            'clip_path': root_path / 'clip' / 'c_clipe'
        }
    }
    model_path = model_name_path_map[model_name]
    if 'clip_path' not in model_path:
        image_all_version = list(model_path['image_path'].glob('*val_acc*'))
        text_all_version = list(model_path['text_path'].glob('*val_acc*'))
        all_version = []
        for i in image_all_version:
            for t in text_all_version:
                all_version.append((i, t))
        return all_version

    else:
        all_version = list(model_path['clip_path'].glob('*val_acc*'))
        return all_version


def load_version(version_path, model_name, device, use_fp16):
    if len(version_path) == 1:
        clip_model = get_clip_model(device, clip_path=version_path, use_fp16=use_fp16)
        return TestCLIPModel(model_name, clip_model, device)
    else:
        clip_model = get_clip_model(device, use_fp16, image_path=version_path[0], text_path=version_path[1])
        return TestCLIPModel(model_name, clip_model, device)


