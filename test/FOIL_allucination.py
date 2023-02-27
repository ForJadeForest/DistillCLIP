import collections
import json
import torch
import clip_score
from sklearn import metrics

from utils import get_model, get_args, get_all_metrics, total_ex


# extract reference
def preprocessCOCO():
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


def computeAccOfFOIL(model, device, mode, cal_other_metric=False):
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

    with torch.autocast('cuda' if 'cuda' in device else 'cpu'):
        _, clip_scores, _, ref_clip_res, = clip_score.get_all_clip_score(model, images, refs, candidates, device)

    clip_labels = []
    for num, clip_result in enumerate(clip_scores):
        if (num % 2) == 0:
            if clip_result >= clip_scores[num + 1]:
                clip_labels.append(1)
            else:
                clip_labels.append(0)
    acc = metrics.accuracy_score(labels, clip_labels)
    print("clipscore {} accuracy is ".format(mode), acc)

    refClip_labels = []
    for num, refclip_result in enumerate(ref_clip_res):
        if (num % 2) == 0:
            if refclip_result >= clip_scores[num + 1]:
                refClip_labels.append(1)
            else:
                refClip_labels.append(0)
    ref_acc = metrics.accuracy_score(labels, refClip_labels)
    print("Refclipscore accuracy is ", ref_acc)

    res_dict = {
        'CLIP-S-acc': round(100 * acc, 2),
        'RefCLIP-S-acc': round(100 * ref_acc, 2)
    }

    if not cal_other_metric:
        return res_dict

    other_metrics = get_all_metrics(refs, candidates, return_per_cap=False)
    for k, v in other_metrics.items():
        if k == 'bleu':
            v = v[-1]  # just do BLEU-4
            k = 'bleu-4'
        if k == 'spice':
            v = [float(item['All']['f']) for item in v]
        metric_labels = []
        for num, result in enumerate(v):
            if (num % 2) == 0:
                if result > v[num + 1]:
                    metric_labels.append(1)
                else:
                    metric_labels.append(0)
        acc = metrics.accuracy_score(labels, metric_labels)
        res_dict[k] = acc
        print("{} accuracy is ".format(k), acc)
    return res_dict


def main(args):
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    model = get_model(device, load_teacher, clip_path, image_path, text_path)
    # 1:1 ref 0:4 ref
    res_dict = {}
    for mode in [0, 1]:
        res = computeAccOfFOIL(model, device, mode, args.cal_other_metric)
        res_dict[f'FOIL_mode_{mode}'] = res
    return res_dict


if __name__ == '__main__':
    args = get_args()
    total_ex(args, main)
