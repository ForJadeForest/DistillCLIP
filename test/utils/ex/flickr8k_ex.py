'''
Computes the metrics for Flickr8K.
'''

from scipy import stats
import os
import json
import numpy as np


def load_data(input_json_path, image_directory):
    data = {}
    with open(input_json_path) as f:
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
    return images, refs, candidates, human_scores


def compute_human_correlation(model, input_json, image_directory, tauvariant='c'):
    """
     root_dir = '/data/pyz/data/flickr8k'
    :param model:
    :param input_json:
    :param image_directory:
    :param tauvariant:
    :return:
    """
    images, refs, candidates, human_scores = load_data(input_json, image_directory)

    final_result = {}
    score_dict = model(images, refs, candidates, reduction=False)
    for score_name, score in score_dict:
        tau = 100 * stats.kendalltau(score, human_scores, variant=tauvariant)[0]
        final_result[score_name] = round(tau, 2)
        print(f'{score_name} Tau-{tauvariant}: {final_result[score_name]}')
    return final_result


def flickr8k_expert_ex(model, root_dir):
    flickr8k_expert_file = os.path.join(root_dir, 'flickr8k.json')
    return compute_human_correlation(model, flickr8k_expert_file, image_directory=root_dir, tauvariant='c')


def flickr8k_cf_ex(model, root_dir):
    flickr8k_crowdflower_file = os.path.join(root_dir, 'crowdflower_flickr8k.json')
    return compute_human_correlation(model, flickr8k_crowdflower_file, image_directory=root_dir, tauvariant='c')
