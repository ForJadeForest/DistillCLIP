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

    score_dict = model(images, candidates, refs, reduction=False)
    return score_dict, np.array(human_scores), images, refs, candidates


def flickr8k_expert_ex(model, root_dir):
    flickr8k_expert_file = os.path.join(root_dir, 'flickr8k.json')
    return compute_human_correlation(model, flickr8k_expert_file, image_directory=root_dir, tauvariant='c')


def flickr8k_cf_ex(model, root_dir):
    flickr8k_crowdflower_file = os.path.join(root_dir, 'crowdflower_flickr8k.json')
    return compute_human_correlation(model, flickr8k_crowdflower_file, image_directory=root_dir, tauvariant='b')


from test.utils.load_model import load_model

clip_model = load_model('L-CLIP', device='cuda:1')
cider_model = load_model('cider')
flickr8k_root_dir = '/data/pyz/data/flickr8k'
clip_res, human_scores, images, refs, candidates = flickr8k_expert_ex(clip_model, '/data/pyz/data/flickr8k')
cider_score, _, _, _, _ = flickr8k_expert_ex(cider_model, '/data/pyz/data/flickr8k')


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


clip_score = standardization(clip_res['score'])
ref_clip_score = standardization(clip_res['ref_score'])
cider_score = standardization(cider_score['ref_score'])
human_scores = standardization(human_scores)

# 1. cider 高，但是human rating 低
# 比如 human 0.2 clip 0.3 cider 0.7
# cider - human_rating > margin
# print clip score
margin = 4
res = []
for i, (cider, h_r) in enumerate(zip(cider_score, human_scores)):
    if cider - h_r > margin and h_r < 1:
        res.append(i)
        print(cider, h_r, clip_score[i], ref_clip_score[i])
        print(images[i])
        print(candidates[i])
        print(refs[i])
        print('=' * 20)

# 2. cider 低，但是human rating 高

margin = 3
res = []
for i, (cider, h_r) in enumerate(zip(cider_score, human_scores)):
    if h_r - cider > margin and clip_score[i] > 2:
        res.append(i)
        print(cider, h_r, clip_score[i], ref_clip_score[i])
        print(images[i])
        print(candidates[i])
        print(refs[i])
        print('=' * 20)


import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
import seaborn as sns

# f, ax1 = plt.subplots(1, 1, figsize=(8, 6))
# c1, c2, c3 = sns.color_palette('Set1', 3)

sns.kdeplot(clip_score, fill=True,  label='clip')
sns.kdeplot(cider_score, fill=True,  label='cider')
sns.kdeplot(human_scores, fill=True,  label='human rating')
import pickle
with open('data.pkl', 'w')as f:
    pickle.dump({
        'clip_score':  clip_score,
        'cider_score': cider_score,
        'human_rating': human_scores
    }, f)
plt.legend()
plt.show()