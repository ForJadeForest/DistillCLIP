import hashlib
import os
import urllib
import warnings
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    return list(_MODELS.keys())


def load(name: str, download_root: str = None):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        model = torch.jit.load(opened_file, map_location="cpu").eval()
    return model.state_dict()


def get_transformer_para(state_dict):
    transformer_para = {
        'embed_dim': state_dict["text_projection"].shape[1],
        'context_length': state_dict["positional_embedding"].shape[0],
        'vocab_size': state_dict["token_embedding.weight"].shape[0],
        'transformer_width': state_dict["ln_final.weight"].shape[0],
        'transformer_heads': state_dict["ln_final.weight"].shape[0] // 64,
        'transformer_layers': len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks"))),
    }
    return transformer_para


def get_visual_transformer_para(state_dict):
    vit = "visual.proj" in state_dict
    embed_dim = state_dict["text_projection"].shape[1]
    if vit:
        print('get the parameters of visual transformer')
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        vision_heads = vision_width // 64

        image_encoder_para = {
            'layers': vision_layers,
            'width': vision_width,
            'patch_size': vision_patch_size,
            'input_resolution': image_resolution,
            'heads': vision_heads,
            'output_dim': embed_dim
        }
    else:
        raise ValueError('the state_dict is error, you should give the state_dict of clip model')
    return image_encoder_para


def teacher_load(teacher_name: str, download_root, model_type, need_layers=None):
    from .component.text_encoder import TextEncoder
    from .component.image_encoder import ImageEncoder
    state_dict = load(teacher_name, download_root=download_root)
    if model_type == 'text':
        para = get_transformer_para(state_dict)
        para.update(dict(need_layers=need_layers))
        teacher_model = TextEncoder(is_student=False, **para)
        my_state_dict = teacher_model.state_dict()
        for k in my_state_dict:
            if k in state_dict:
                my_state_dict[k] = state_dict[k]
        teacher_model.load_state_dict(my_state_dict)
        return teacher_model
    elif model_type == 'image':
        para = get_visual_transformer_para(state_dict)
        para.update(dict(need_layers=need_layers))
        teacher_model = ImageEncoder(is_student=False, vit_paras=para)
        my_state_dict = teacher_model.state_dict()
        for k in my_state_dict:
            if k in state_dict:
                my_state_dict[k] = state_dict[k]
        teacher_model.load_state_dict(my_state_dict)
        return teacher_model
    elif model_type == 'all':
        from .component.clip_model import CLIPModel
        vit_paras = get_visual_transformer_para(state_dict)
        vit_paras.update(dict(need_layers=need_layers))
        trans_para = get_transformer_para(state_dict)
        trans_para.update(dict(need_layers=need_layers))
        image_encoder = ImageEncoder(is_student=False, vit_paras=vit_paras)
        text_encoder = TextEncoder(is_student=False, **trans_para)
        teacher_model = CLIPModel(False, image_encoder, text_encoder)
        my_state_dict = teacher_model.state_dict()
        map_d = {
            k: False for k in my_state_dict.keys()
        }
        for k in my_state_dict:
            if k in state_dict:
                my_state_dict[k] = state_dict[k]
                map_d[k] = True
            elif k.startswith('text_encoder'):
                my_state_dict[k] = state_dict[k.replace('text_encoder.', '')]
                map_d[k] = True
            elif k.startswith('image_encoder'):
                my_state_dict[k] = state_dict[k.replace('image_encoder.', '')]
                map_d[k] = True
        teacher_model.load_state_dict(my_state_dict)
        return teacher_model
