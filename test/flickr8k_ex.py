'''
Computes the metrics for Flickr8K.
'''
import sys

sys.path.append('../')
import clip_score
import generation_eval_utils
import scipy.stats
import os
import json
import numpy as np
import torch
import warnings
from test.utils import get_model


def compute_human_correlation(model, device, input_json, image_directory, tauvariant='c'):
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
    other_metrics = generation_eval_utils.get_all_metrics(refs, candidates, return_per_cap=True)

    print('CLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                            scipy.stats.kendalltau(per_instance_image_text, human_scores,
                                                                   variant=tauvariant)[0]))
    print('RefCLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                               scipy.stats.kendalltau(refclipscores, human_scores, variant=tauvariant)[
                                                   0]))

    for k, v in other_metrics.items():
        if k == 'bleu':
            v = v[-1]  # just do BLEU-4
            k = 'bleu-4'
        if k == 'spice':
            v = [float(item['All']['f']) for item in v]

        print('{} Tau-{}: {:.3f}'.format(k, tauvariant,
                                         100 * scipy.stats.kendalltau(v, human_scores, variant=tauvariant)[0]))


def main():
    image_directory = '/data/pyz/data/flickr8k'

    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')

    print('Now is the Distillation model')
    image_path = '/home/pyz32/code/Dis_CLIP/test/test_checkpoint/image/ws_best/234-val_acc0.262-loss0.11146.ckpt'
    text_path = '/home/pyz32/code/Dis_CLIP/CLIPDistillation/k0n6x46s/checkpoints/85-val_acc0.301-loss0.03612.ckpt'
    model = get_model(device, image_path, text_path)

    if not os.path.exists('test_dataset/flickr8k/flickr8k.json'):
        print('Please run download.py')
        quit()
    print('Flickr8K Expert (Tau-c)')
    flickr8k_expert_file = os.path.join(image_directory, 'flickr8k.json')
    compute_human_correlation(model, device, flickr8k_expert_file, image_directory, tauvariant='c')

    print('Flickr8K CrowdFlower (Tau-b)')
    flickr8k_crowdflower_file = os.path.join(image_directory, 'crowdflower_flickr8k.json')
    compute_human_correlation(model, device, flickr8k_crowdflower_file, image_directory, tauvariant='b')

    print('Now is the Original model')
    del model
    model = get_model(device)

    print('Flickr8K Expert (Tau-c)')
    flickr8k_expert_file = os.path.join(image_directory, 'flickr8k.json')
    compute_human_correlation(model, device, flickr8k_expert_file, image_directory, tauvariant='c')

    print('Flickr8K CrowdFlower (Tau-b)')
    flickr8k_crowdflower_file = os.path.join(image_directory, 'crowdflower_flickr8k.json')
    compute_human_correlation(model, device, flickr8k_crowdflower_file, image_directory, tauvariant='b')


if __name__ == '__main__':
    main()