'''
Computes the metrics for composite.
image_path!

'''
import sys

sys.path.append('../')
import clip_score
import utils
# from utils import get_all_metrics
import scipy.stats
import os
import json
import numpy as np
import torch
import warnings
import clip


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


    '''device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()'''
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
    

    print('CLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                            scipy.stats.kendalltau(per_instance_image_text, human_scores,
                                                                   variant=tauvariant)[0]))
    print('Only-ref CLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                            scipy.stats.kendalltau(per_instance_text_text, human_scores,
                                                                   variant=tauvariant)[0]))

    print('RefCLIPScore Tau-{}: {:.3f}'.format(tauvariant, 100 *
                                               scipy.stats.kendalltau(refclipscores, human_scores, variant=tauvariant)[
                                                   0]))

    other_metrics = utils.get_all_metrics(refs, candidates, return_per_cap=True)
    for k, v in other_metrics.items():
        if k == 'bleu':
            v = v[-1]  # just do BLEU-4
            k = 'bleu-4'
        if k == 'spice':
            v = [float(item['All']['f']) for item in v]

        print('{} Tau-{}: {:.3f}'.format(k, tauvariant,
                                         100 * scipy.stats.kendalltau(v, human_scores, variant=tauvariant)[0]))


def main():
    if not os.path.exists("/data/ll/composite/composite.json"):
        print('Please run composite_preprocess.py')
        quit()
    print('composite (Tau-c)')

    args = utils.get_args()
    device = args.device
    clip_model = utils.get_model(device, use_fp16= args.fp16)

    if args.use_origin:
        print("=" * 10 + "composite Tau-c; Using model: origin" + "=" * 10)
        compute_human_correlation("/data/ll/composite/composite.json", clip_model, device,  tauvariant='c')

    image_path = args.image_path
    text_path = args.text_path
    clip_model = utils.get_model(device, image_path, text_path, use_fp16=args.fp16)
    print("=" * 10 + "composite Tau-c; Using model: distilled" + "=" * 10)
    compute_human_correlation("/data/ll/composite/composite.json", clip_model, device, tauvariant='c')

if __name__ == '__main__':
    main()