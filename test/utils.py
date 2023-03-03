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

# Model_Type_List = ['baseline', 'smd', 'compression_freeze_text', 'cos_diff', 'compression_text', 'teacher']
Model_Type_List = ['baseline', 'freeze_text', 'compression_text', 'compression_freeze_text', 'ce_loss_clip', 'teacher']
# Model_Type_List = ['baseline', 'compression_text', 'teacher']
# Model_Type_List = ['teacher']










def load_compression_text_clip_model(args):
    print("[INFO] ==> Now load the compression_text clip model!")
    # args.image_path = '/data/share/pyz/Dis_CLIP/final/image/ws_no_smd/174-val_acc0.243-loss0.13381.ckpt'
    # args.text_path = '/data/share/pyz/Dis_CLIP/final/text/compression/201-val_acc0.299-loss0.04052.ckpt'

    args.image_path = '/data/share/pyz/Dis_CLIP/final/image/ws_no_smd/188-val_acc0.237-loss0.13170.ckpt'
    args.text_path = '/data/share/pyz/Dis_CLIP/final/text/compression/135-val_acc0.299-loss0.04125.ckpt'

    args.clip_path = None
    args.load_teacher = False
    return args


def load_ce_loss_clip_model(args):
    print("[INFO] ==> Now load the freeze text cos_diff clip model!")
    args.clip_path = '/data/share/pyz/Dis_CLIP/final/clip/ce_loss/checkpoints/290-val_acc0.242-loss0.28316.ckpt'
    args.image_path = None
    args.text_path = None
    args.load_teacher = False
    return args


def load_freeze_text_cos_diff_clip_model(args):
    print("[INFO] ==> Now load the freeze text cos_diff clip model!")
    args.clip_path = '/data/share/pyz/Dis_CLIP/final/clip/freeze_text/117-val_acc0.249-loss0.05607.ckpt'
    args.image_path = None
    args.text_path = None
    args.load_teacher = False
    return args


def load_compression_freeze_text_cos_diff_clip_model(args):
    print("[INFO] ==> Now load the compression freeze text cos_diff clip model!")
    args.clip_path = '/data/share/pyz/Dis_CLIP/final/clip/compression_text/253-val_acc0.245-loss0.05582-v1.ckpt'
    args.image_path = None
    args.text_path = None
    args.load_teacher = False
    return args


def load_smd_clip_model(args):
    print("[INFO] ==> Now load the smd clip model!")
    args.image_path = '/data/share/pyz/Dis_CLIP/final/image/ws_best/234-val_acc0.262-loss0.11146.ckpt'
    args.text_path = '/data/share/pyz/Dis_CLIP/final/text/ws_best/225-val_acc0.301-loss0.03477.ckpt'
    args.clip_path = None
    args.load_teacher = False
    return args


def load_baseline_clip_model(args):
    print("[INFO] ==> Now load the baseline clip model!")
    # args.image_path = '/data/share/pyz/Dis_CLIP/final/image/ws_no_smd/174-val_acc0.243-loss0.13381.ckpt'
    # args.text_path = '/data/share/pyz/Dis_CLIP/final/text/ws_no_smd/131-val_acc0.300-loss0.03917.ckpt'

    args.image_path = '/data/share/pyz/Dis_CLIP/final/image/ws_no_smd/188-val_acc0.237-loss0.13170.ckpt'
    args.text_path = '/data/share/pyz/Dis_CLIP/final/text/ws_no_smd/163-val_acc0.296-loss0.03877.ckpt'

    args.clip_path = None
    args.load_teacher = False
    return args


def load_original_clip(args):
    print("[INFO] ==> Now load the original clip model!")
    args.image_path = None
    args.text_path = None
    args.clip_path = None
    args.load_teacher = True
    return args


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--image_path', type=str, help='The image encoder checkpoint path',
                       default=None)
    parse.add_argument('-t', '--text_path', type=str, help='The text encoder checkpoint path',
                       default=None)
    parse.add_argument('--clip_path', type=str, help='The CLIP model checkpoint path',
                       default=None)
    parse.add_argument('--model_type', type=str, help='load the model type', default='smd')
    parse.add_argument('-d', '--device', type=str, help='The device(Gpu or Cpu)', default='cuda')
    parse.add_argument('-p', '--fp16', action='store_true', help='Whether use the fp16', default=True)
    parse.add_argument('--load_teacher', action='store_true',
                       help='Whether use the origin clip model to do test', default=False)
    parse.add_argument('--cal_other_metric', action='store_true',
                       help='Whether use the origin clip model to do test', default=False)
    parse.add_argument('--model_name', type=str,
                       help='Whether use the origin clip model to do test', default=False)
    args = parse.parse_args()
    return args


def change_args(args, model_type):
    if model_type == 'baseline':
        args = load_baseline_clip_model(args)
    elif model_type == 'smd':
        args = load_smd_clip_model(args)
    elif model_type == 'teacher':
        args = load_original_clip(args)
    elif model_type == 'compression_text':
        args = load_compression_text_clip_model(args)
    elif model_type == 'freeze_text':
        args = load_freeze_text_cos_diff_clip_model(args)
    elif model_type == 'compression_freeze_text':
        args = load_compression_freeze_text_cos_diff_clip_model(args)
    elif model_type == 'ce_loss_clip':
        args = load_ce_loss_clip_model(args)
    else:
        raise ValueError(f'the model_type should in {Model_Type_List}, instead of {args.model_type}')
    return args


def total_ex(args_control, ex_function, *args, **kwargs):
    model_type_res = {}
    # 对每一实验，需要返回一个字典：key是model_type，value是一个字典（key是指标，value该指标对应的值）
    # 但是每一个脚本可能返回多个value这样的字典，每一个实验中有小实验
    # 也就是需要一个这样的字典：
    # {model_type: {sub_ex_1: {metric1: value1, metric2: value2}, sub_ex_2: {{metric1: value1}}}
    with torch.no_grad():
        for model_type in Model_Type_List:
            print('=' * 10 + f'[INFO] ==> begin Test {model_type} CLIP Model' + '=' * 10)
            args_control = change_args(args_control, model_type)
            args_control.model_name = model_type
            res_dict = ex_function(args_control, *args, **kwargs)
            print('*==*' * 20)
            model_type_res[model_type] = res_dict
    ex_name_list = model_type_res[Model_Type_List[0]].keys()
    final_res = {}
    for ex_name in ex_name_list:
        final_res[ex_name] = {model_type: model_type_res[model_type][ex_name] for model_type in Model_Type_List}
    import pandas as pd
    for ex_name in ex_name_list:
        ex_res_df = pd.DataFrame(final_res[ex_name])
        ex_res_df.to_csv(f'./result_2/{ex_name}.csv')


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
