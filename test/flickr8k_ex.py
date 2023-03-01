'''
Computes the metrics for Flickr8K.
'''
import sys

sys.path.append('../')
import clip_score
import scipy.stats
import os
import json
import numpy as np
import torch
from test.utils import get_model, get_all_metrics, total_ex, get_args


def compute_human_correlation(model, device, input_json, image_directory, tauvariant='c'):
    final_result = {}
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
            images.append(image_directory + '/' + v['image_path'].replace('Flicker8k', 'Flickr8k'))
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))
            human_scores.append(human_judgement['rating'])

    with torch.autocast('cuda'):
        image_feats = clip_score.extract_all_images(
            images, model, device, batch_size=64, num_workers=8)

        # get image-text clipscore
        _, per_instance_image_text, candidate_feats = clip_score.get_clip_score(
            model, image_feats, candidates, device)

        # get text-text clipscore
        _, per_instance_text_text = clip_score.get_refonlyclipscore(
            model, refs, candidate_feats, device)

    # F-score
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (
            per_instance_image_text + per_instance_text_text)

    tau = 100 * scipy.stats.kendalltau(per_instance_image_text, human_scores, variant=tauvariant)[0]
    final_result['clip_score_kenalltau'] = round(tau, 2)
    tau = 100 * scipy.stats.kendalltau(refclipscores, human_scores, variant=tauvariant)[0]
    final_result['ref_clip_score'] = round(tau, 2)

    print('CLIPScore Tau-{}: {:.3f}'.format(tauvariant, final_result['clip_score_kenalltau']))
    print('RefCLIPScore Tau-{}: {:.3f}'.format(tauvariant, final_result['ref_clip_score']))

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


def flickr8k_ex(model, device, root_dir):
    final_res = {}

    print('Flickr8K Expert (Tau-c)')
    flickr8k_expert_file = os.path.join(root_dir, 'flickr8k.json')
    final_res['Flickr8K-Expert'] = compute_human_correlation(model, device, flickr8k_expert_file, root_dir,
                                                             tauvariant='c')

    print('Flickr8K CrowdFlower (Tau-b)')
    flickr8k_crowdflower_file = os.path.join(root_dir, 'crowdflower_flickr8k.json')
    final_res['Flickr8K-CF'] = compute_human_correlation(model, device, flickr8k_crowdflower_file, root_dir,
                                                         tauvariant='b')

    return final_res


def main(args):
    root_dir = '/data/pyz/data/flickr8k'
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    model = get_model(device, load_teacher, clip_path, image_path, text_path)
    return flickr8k_ex(model, device, root_dir)


if __name__ == '__main__':
    args = get_args()
    total_ex(args, main)
