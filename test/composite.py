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


def compute_human_correlation(input_json, model, device, tauvariant='c'):
    data = {}
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

    with torch.autocast('cuda'):
        image_feats = clip_score.extract_all_images(
            images, model, device, batch_size=2048, num_workers=8)

        # get image-text clipscore
        _, per_instance_image_text, candidate_feats = clip_score.get_clip_score(
            model, image_feats, candidates, device)

        # get text-text clipscore
        _, per_instance_text_text = clip_score.get_refonlyclipscore(
            model, refs, candidate_feats, device)

    # F-score
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (
            per_instance_image_text + per_instance_text_text)

    print('CLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                            scipy.stats.kendalltau(per_instance_image_text, human_scores,
                                                                   variant=tauvariant)[0]))
    print('Only-ref CLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                                     scipy.stats.kendalltau(per_instance_text_text, human_scores,
                                                                            variant=tauvariant)[0]))

    print('RefCLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                               scipy.stats.kendalltau(refclipscores, human_scores, variant=tauvariant)[
                                                   0]))

    other_metrics = get_all_metrics(refs, candidates, return_per_cap=True)
    for k, v in other_metrics.items():
        if k == 'bleu':
            v = v[-1]  # just do BLEU-4
            k = 'bleu-4'
        if k == 'spice':
            v = [float(item['All']['f']) for item in v]

        print('{} Tau-{}: {:.3f}'.format(k, tauvariant,
                                         100 * scipy.stats.kendalltau(v, human_scores, variant=tauvariant)[0]))


def composite_ex(clip_model, device, root_dir):
    print('composite (Tau-c)')
    composite_file = os.path.join(root_dir, 'composite.json')
    if not os.path.exists(composite_file):
        print('Please run composite_preprocess.py')
        quit()
    compute_human_correlation(composite_file, clip_model, device, tauvariant='c')


def main(args):
    root_dir = '/data/ll/composite'
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    model = get_model(device, load_teacher, clip_path, image_path, text_path)
    composite_ex(model, device, root_dir)


if __name__ == '__main__':
    args = get_args()
    total_ex(args, main)
