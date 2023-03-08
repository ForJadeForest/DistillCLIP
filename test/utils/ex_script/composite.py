from scipy import stats
import os
import json
import numpy as np


def load_data(input_json_path):
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
            images.append(v['image_path'])  # image_path need to be changed in preprocess module
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))
            human_scores.append(human_judgement['rating'])
    return images, refs, candidates, human_scores


def compute_human_correlation(input_json, model, tauvariant='c'):
    images, refs, candidates, human_scores = load_data(input_json)

    final_result = {}
    score_dict = model(images, candidates, refs,  reduction=False)
    for score_name, score in score_dict.items():
        tau = 100 * stats.kendalltau(score, human_scores, variant=tauvariant)[0]
        final_result[score_name] = round(tau, 2)
        print(f'{score_name} Tau-{tauvariant}: {final_result[score_name]}')
    return final_result


def composite_ex(model, root_dir):
    """
    :param model:
    :param root_dir:root_dir = '/data/ll/composite'
    :return:
    """
    composite_file = os.path.join(root_dir, 'composite.json')
    if not os.path.exists(composite_file):
        print('Please run composite_preprocess.py')
        quit()

    return compute_human_correlation(composite_file, model, tauvariant='c')
