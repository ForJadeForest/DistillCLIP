import argparse

import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from model.component.clip_model import CLIPModel
from model.component.weight_share_model import RepeatVisionTransformer, RepeatTextTransformer
from model.utils import teacher_load


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


# Path list:
# /data/share/pyz/Dis_CLIP/final/image/ws_best/234-val_acc0.262-loss0.11146.ckpt  (smd image)
# /data/share/pyz/Dis_CLIP/final/image/ws_no_smd/174-val_acc0.243-loss0.13381.ckpt   (no smd image)
# /data/share/pyz/Dis_CLIP/final/text/ws_best/225-val_acc0.301-loss0.03477.ckpt  (smd text)
# /data/share/pyz/Dis_CLIP/final/text/ws_text_no_smd/131-val_acc0.300-loss0.03917.ckpt (no smd text)

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--image_path', type=str, help='The image encoder checkpoint path',
                       default='/data/share/pyz/Dis_CLIP/final/image/ws_no_smd/174-val_acc0.243-loss0.13381.ckpt')
    parse.add_argument('-t', '--text_path', type=str, help='The text encoder checkpoint path',
                       default='/data/share/pyz/Dis_CLIP/final/text/ws_text_no_smd/131-val_acc0.300-loss0.03917.ckpt')
    parse.add_argument('-d', '--device', type=str, help='The device(Gpu or Cpu)', default='cuda')
    parse.add_argument('-p', '--fp16', action='store_true', help='Whether use the fp16', default=True)
    parse.add_argument('-o', '--use_origin', action='store_true',
                       help='Whether use the origin clip model to do test', default=False)
    return parse.parse_args()


def get_all_metrics(refs, cands, return_per_cap=False):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(4), 'bleu'),
                               (Meteor(), 'meteor'),
                               (Rouge(), 'rouge'),
                               (Cider(), 'cider'),
                               (Spice(), 'spice')]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        if return_per_cap:
            metrics.append(per_cap)
        else:
            metrics.append(overall)
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics


def tokenize(refs, cands, no_op=False):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {idx: [{'caption': r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption': c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


def pycoco_eval(scorer, refs, cands):
    '''
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    '''
    refs, cands = tokenize(refs, cands)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores
