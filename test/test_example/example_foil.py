import collections
import json
from sklearn import metrics
import numpy as np

from test.utils.load_model import load_model


def preprocessCOCO():
    # extract reference
    with open("/data/ll/composite/captions_val2014.json") as f:
        data = {}
        data.update(json.load(f))
        mscoco_image2ann = collections.defaultdict(list)
        for line in data['annotations']:
            mscoco_image2ann[line['image_id']].append(line['caption'])
    print("coco is read")
    return mscoco_image2ann  # {image_id : [references]}


def preprocessFOIL():
    foil_set = json.load(open("/data/ll/composite/foilv1.0_test_2017.json"))
    # testing_y = [0 if f['foil_word'] == 'ORIG' else 1 for f in foil_set['annotations']]
    all_index = collections.defaultdict(list)
    for line in foil_set["annotations"]:
        all_index[line["image_id"]].append([str(line["foil"]), line["caption"], line["id"]])
    print("foil is read")
    return all_index  # {image_id: [['False',"",foil_id],['True',"",foil_id]]} foil_id == pair


def load_data(mode):
    images = []
    refs = []
    candidates = []
    labels = []

    ref_set = preprocessCOCO()
    pair_set = preprocessFOIL()

    raw_images = list(pair_set.keys())

    for image_id in raw_images:
        for caption in pair_set[image_id]:
            if caption[0] == 'False':
                ref = []
                for gt in ref_set[image_id]:
                    if caption[1] == gt:
                        continue
                    else:
                        ref.append(gt)
                if mode == 0:
                    refs.append([' '.join(gt.split()) for gt in ref])
                    refs.append([' '.join(gt.split()) for gt in ref])
                else:
                    refs.append([' '.join(ref[2].split())])
                    refs.append([' '.join(ref[2].split())])
                labels.append(1)
            images.append('/data/pyz/data/mscoco/val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg')
            candidates.append(' '.join(caption[1].split()))
    return images, refs, candidates, labels


def cal_acc(scores):
    pred_label = []
    for num, clip_result in enumerate(scores):
        if (num % 2) == 0:
            if clip_result > scores[num + 1]:
                pred_label.append(1)
            else:
                pred_label.append(0)
    return pred_label


def FOIL_ex(model, mode):
    images, refs, candidates, labels = load_data(mode)
    score_dict = model(images, candidates, refs, reduction=False)
    pred = {}
    if model.is_clip:
        preds = cal_acc(score_dict['score'])
        pred['score'] = preds
    pred['ref_score'] = cal_acc(score_dict['ref_score'])

    return score_dict, pred, images, refs, candidates, labels


clip_model = load_model('L-CLIP', device='cuda:1')
cider_model = load_model('cider')

clip_score, clip_pred, images, refs, candidates, labels = FOIL_ex(clip_model, 0)
cider_score, cider_pred, _, _, _, _ = FOIL_ex(cider_model, 0)

# 1. cider 高，但是human rating 低
# 比如 human 0.2 clip 0.3 cider 0.7
# cider - human_rating > margin
# print clip score

res = []
for i, (cider, clip) in enumerate(zip(cider_pred['ref_score'], clip_pred['score'])):

    if cider != clip and clip == 1 and clip_pred['ref_score'][i] == 1 and cider_score['ref_score'][2 * i + 1] - cider_score['ref_score'][2 * i] > 0.5:
        res.append(i)
        print(cider, clip)
        print('True Score')
        print(cider_score['ref_score'][2 * i], clip_score['score'][2 * i], clip_score['ref_score'][2 * i])
        print('False Score')
        print(cider_score['ref_score'][2 * i + 1], clip_score['score'][2 * i + 1], clip_score['ref_score'][2 * i +  + 1])
        print(images[2 * i])
        print(refs[2 * i])
        print('True Candidate: ', candidates[2 * i])
        print('False Candidate: ', candidates[2 * i + 1])
        print('=' * 20)
