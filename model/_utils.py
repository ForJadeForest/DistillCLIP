import hashlib
import os
import urllib
import warnings
from typing import List

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as f
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
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


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


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


def teacher_load(teacher_name: str, download_root, model_type):
    from .text_encoder import TextEncoder
    from .image_encoder import ImageEncoder
    if model_type == 'text':
        state_dict = load(teacher_name, download_root=download_root)
        para = get_transformer_para(state_dict)
        teacher_model = TextEncoder(is_student=False, **para)

        my_state_dict = teacher_model.state_dict()
        for k in my_state_dict:
            if k in state_dict:
                my_state_dict[k] = state_dict[k]
        teacher_model.load_state_dict(my_state_dict)
        return teacher_model, para['transformer_layers']
    elif model_type == 'image':
        state_dict = load(teacher_name, download_root=download_root)
        para = get_visual_transformer_para(state_dict)
        teacher_model = ImageEncoder(is_student=False, vit_paras=para)
        my_state_dict = teacher_model.state_dict()
        for k in my_state_dict:
            if k in state_dict:
                my_state_dict[k] = state_dict[k]
        teacher_model.load_state_dict(my_state_dict)
        return teacher_model, para['layers']


def output_filter(is_student, representations, embedding_projection, embedding, attention_maps, last_state,
                  hidden_projection, attention_probs, value_maps, need_emb, need_rep, need_attn_score, no_trans):
    if is_student and not no_trans:

        for i in range(len(representations)):
            if need_rep:
                representations[i] = hidden_projection(representations[i])
            else:
                break
        if need_emb:
                embedding = embedding_projection(embedding)
        else:
            embedding = None
    if need_attn_score:
        for i in range(len(attention_maps)):
            attention_maps[i] = torch.where(attention_maps[i] == float('-inf'), torch.zeros_like(attention_maps[i]),
                                            attention_maps[i])
    return last_state, attention_maps, representations, embedding, attention_probs, value_maps


class LayerMap:
    def __init__(self, stu_total_layer_num, tea_total_layer_num, map_type):
        self.stu_total_layer_num = stu_total_layer_num
        self.tea_total_layer_num = tea_total_layer_num
        self.map_type = map_type
        if self.map_type == 'mid':
            assert self.tea_total_layer_num % self.stu_total_layer_num == 0
            self.step = self.tea_total_layer_num // self.stu_total_layer_num

    def __call__(self, stu_layer_num):
        if self.map_type == 'mid':
            return stu_layer_num * self.step
        elif self.map_type == 'begin':
            return stu_layer_num
        elif self.map_type == 'end':
            return self.tea_total_layer_num - self.stu_total_layer_num + stu_layer_num


class LossControl:
    def __init__(self, loss_name, loss_scale: dict = None, temperature=None, percent=None):
        self.loss_name = loss_name
        self.loss_scale = loss_scale
        if self.loss_scale is None:
            self.loss_scale = {n: 1 for n in self.loss_name}
        self.loss = self._init_loss()
        self.temperature = temperature
        self.percent = percent
        if self.percent is None:
            self.percent = {n: 1 / len(loss_name) for n in loss_name}
        assert abs(sum([v for v in self.percent.values()]) - 1) <= 1e-5
        print(self.percent)
        print(self.loss_scale)

    def _init_loss(self):
        losses = {}
        for n in self.loss_name:
            if n == 'l1':
                loss_function = nn.L1Loss()
            elif n == 'ce':
                loss_function = nn.CrossEntropyLoss(reduction='mean')
            elif n == 'kl':
                loss_function = nn.KLDivLoss(reduction='batchmean')
            elif n == 'cos':
                loss_function = nn.CosineEmbeddingLoss()
            elif n == 'emb':
                loss_function = nn.MSELoss()
            elif n == 'attn':
                loss_function = nn.MSELoss()
            elif n == 'attn_probs':
                loss_function = nn.MSELoss()
            elif n == 'hidden':
                loss_function = nn.MSELoss()
            elif n == 'last_attn':
                loss_function = nn.KLDivLoss(reduction='batchmean')
            elif n == 'last_value_map':
                loss_function = nn.KLDivLoss(reduction='batchmean')
            else:
                raise ValueError("Invalid Loss Type!")
            losses[n] = loss_function
        return losses

    def need_output(self):
        need_para = {
            'need_attn_score': False,
            'need_value_map': False,
            'need_attn_prob': False,
            'need_rep': False,
            'need_emb': False
        }
        for n in self.loss_name:
            if n == 'emb':
                need_para['need_emb'] = True
            elif n == 'attn':
                need_para['need_attn_score'] = True
            elif n == 'attn_probs':
                need_para['need_attn_prob'] = True
            elif n == 'hidden':
                need_para['need_rep'] = True
            elif n == 'last_attn':
                need_para['need_attn_prob'] = True
            elif n == 'last_value_map':
                need_para['need_value_map'] = True

        return need_para

    def cal_loss(self, stu_out, tea_out, layer_map: LayerMap, device):
        stu_last_rep, stu_attention_maps, stu_representations, stu_embedding, stu_attention_probs, stu_value_map = stu_out
        tea_last_rep, tea_attention_maps, tea_representations, tea_embedding, tea_attention_probs, tea_value_map = tea_out
        stu_layer_num = layer_map.stu_total_layer_num

        cal_res = {}
        for loss_name in self.loss:
            loss = self.loss[loss_name]
            if loss_name == 'emb':
                # calculate the embedding loss
                cal_res[loss_name] = loss(stu_embedding, tea_embedding)
            elif loss_name == 'attn':
                # attention loss
                attn_loss = 0
                for layer_num, stu_attn_out in enumerate(stu_attention_maps):
                    stu_head_num = stu_attn_out.shape[1]
                    tea_head_num = tea_attention_maps[layer_map(layer_num)].shape[1]
                    stu_mean_head_out = torch.sum(stu_attn_out, dim=1) / stu_head_num
                    tea_mean_head_out = torch.sum(tea_attention_maps[layer_map(layer_num)], dim=1) / tea_head_num
                    attn_loss += loss(stu_mean_head_out, tea_mean_head_out)
                attn_loss /= stu_layer_num
                cal_res[loss_name] = attn_loss
            elif loss_name == 'hidden':
                hidden_loss = 0
                for layer_num, stu_attn_out in enumerate(stu_representations):
                    hidden_loss += loss(stu_representations[layer_num],
                                        tea_representations[layer_map(layer_num)])
                hidden_loss /= stu_layer_num
                cal_res[loss_name] = hidden_loss
            elif loss_name == 'l1':
                logits_l1_loss = loss(stu_last_rep, tea_last_rep)
                cal_res[loss_name] = logits_l1_loss
            elif loss_name == 'kl':
                # calculate the pred loss
                assert self.temperature, 'You should give the temperature for the kl loss'
                logits_ce_loss = loss(
                    f.softmax(stu_last_rep / self.temperature, dim=1).log(),
                    f.softmax(tea_last_rep / self.temperature, dim=1)
                ) * self.temperature ** 2
                cal_res[loss_name] = logits_ce_loss
            elif loss_name == 'cos':
                logits_cos_loss = loss(stu_last_rep, tea_last_rep, torch.ones(len(stu_last_rep), device=device))
                cal_res[loss_name] = logits_cos_loss
            elif loss_name == 'ce':
                logits_ce_loss = loss(
                    stu_last_rep.softmax(dim=1),
                    tea_last_rep.softmax(dim=1)
                )
                cal_res[loss_name] = logits_ce_loss
            elif loss_name == 'last_attn':
                cal_res[loss_name] = loss(
                    f.softmax(stu_attention_probs[-1], dim=1).log(),
                    f.softmax(tea_attention_probs[-1], dim=1)
                )
            elif loss_name == 'last_value_map':
                cal_res[loss_name] = loss(
                    f.softmax(stu_value_map, dim=1).log(),
                    f.softmax(tea_value_map, dim=1)
                )
            elif loss_name == 'attn_probs':
                attn_loss = 0
                for layer_num, stu_attn_out in enumerate(stu_attention_probs):
                    stu_head_num = stu_attn_out.shape[1]
                    tea_head_num = tea_attention_probs[layer_map(layer_num)].shape[1]
                    stu_mean_head_out = torch.sum(stu_attn_out, dim=1) / stu_head_num
                    tea_mean_head_out = torch.sum(tea_attention_probs[layer_map(layer_num)], dim=1) / tea_head_num
                    attn_loss += loss(stu_mean_head_out, tea_mean_head_out)
                attn_loss /= stu_layer_num
                cal_res[loss_name] = attn_loss

        loss = 0
        for (loss_name, scale) in self.loss_scale.items():
            cal_res[loss_name] = cal_res[loss_name] * scale
            loss += cal_res[loss_name] * self.percent[loss_name]

        return loss, cal_res
