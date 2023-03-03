'''
Computes the metrics for composite.
image_path!

'''
import sys

sys.path.append('../')
import clip_score
from utils import get_model, get_args, get_all_metrics, total_ex
import scipy.stats
import os
import json
import numpy as np
import torch


def compute_human_correlation(input_json, model, device, tauvariant='c', cal_other_metric=False):
    data = {}
    final_result = {}
    with open(input_json) as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    images = []
    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue
            images.append(v['image_path'])  # image_path need to be changed in preprocess module
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))
            human_scores.append(human_judgement['rating'])

    with torch.autocast('cuda' if 'cuda' in device else 'cpu'):
        _, clip_scores, _, ref_clip_res, = clip_score.get_all_clip_score(model, images, refs, candidates, device)

    tau = 100 * scipy.stats.kendalltau(clip_scores, human_scores, variant=tauvariant)[0]
    final_result['clip_score_kenalltau'] = round(tau, 2)
    print('CLIPScore Tau-{}: {:.3f}'.format(tauvariant, tau))
    # print('Only-ref CLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
    #                                                  scipy.stats.kendalltau(per_instance_text_text, human_scores,
    #                                                                         variant=tauvariant)[0]))
    tau = 100 * scipy.stats.kendalltau(ref_clip_res, human_scores, variant=tauvariant)[0]
    final_result['ref_clip_score'] = round(tau, 2)
    print('RefCLIPScore Tau-{}: {:.3f}'.format(tauvariant, tau))

    if not cal_other_metric:
        return final_result

    other_metrics = get_all_metrics(refs, candidates, return_per_cap=True)
    for k, v in other_metrics.items():
        if k == 'bleu':
            v = v[-1]  # just do BLEU-4
            k = 'bleu-4'
        if k == 'spice':
            v = [float(item['All']['f']) for item in v]
        tau = 100 * scipy.stats.kendalltau(v, human_scores, variant=tauvariant)[0]
        final_result[k] = round(tau, 2)
        print('{} Tau-{}: {:.3f}'.format(k, tauvariant, tau))
    return final_result


def composite_ex(clip_model, device, root_dir, cal_other_metric):
    final_res = {}
    composite_file = os.path.join(root_dir, 'composite.json')
    if not os.path.exists(composite_file):
        print('Please run composite_preprocess.py')
        quit()
    final_res['Composite'] = compute_human_correlation(composite_file, clip_model, device, tauvariant='c',
                                                       cal_other_metric=cal_other_metric)
    return final_res


def main(args):
    root_dir = '/data/ll/composite'
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    model = get_model(device, load_teacher, clip_path, image_path, text_path)
    return composite_ex(model, device, root_dir, args.cal_other_metric)


if __name__ == '__main__':
    args = get_args()
    total_ex(args, main)
