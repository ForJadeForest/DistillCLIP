import torch
from model.component.clip_model import CLIPModel
from model.utils import teacher_load
from model.component.weight_share_model import RepeatVisionTransformer, RepeatTextTransformer


def load_image_encoder(cpk_path):
    cpk = torch.load(cpk_path)
    visual_encoder = RepeatVisionTransformer(
        img_size=224, patch_size=32, in_chans=3, out_dim=512, embed_dim=768, depth=6, num_heads=24, mlp_ratio=4.0,
        qkv_bias=True, repeated_times=2, use_transform=True
    )
    state_dict = {k.replace('student.', ''): v for k, v in cpk['state_dict'].items() if k.startswith('student')}
    visual_encoder.load_state_dict(state_dict)
    return visual_encoder


def load_text_encoder(cpk_path):
    cpk = torch.load(cpk_path)
    text_encoder = RepeatTextTransformer(
        depth=4, repeated_times=2,
        use_transform=True
    )
    state_dict = {k.replace('student.', ''): v for k, v in cpk['state_dict'].items() if k.startswith('student')}
    text_encoder.load_state_dict(state_dict)
    return text_encoder

def get_model(device, image_path=None, text_path=None, use_fp16=True) -> CLIPModel:
    """
    image_path: the image encoder checkpoint path
    text_path: the text encoder checkpoint path
    device: the device for the model
    """
    if image_path is not None and text_path is not None:
        image_encoder = load_image_encoder(image_path)
        text_encoder = load_text_encoder(text_path)
        clip_model = CLIPModel(False, image_encoder, text_encoder, only_last_rep=True)
    else:
        print("[INFO] ==> Now load the original clip model!")
        clip_model = teacher_load('ViT-B/32', r'/data/pyz/.cache', model_type='all', only_last_rep=True)
    # clip model default use half precision
    if use_fp16:
        clip_model.half()
    clip_model.to(device)
    clip_model.eval()
    clip_model.only_last_rep = True
    return clip_model
