import collections
import json
from sklearn import metrics


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


def cal_acc(scores, labels):
    pred_label = []
    for num, clip_result in enumerate(scores):
        if (num % 2) == 0:
            if clip_result > scores[num + 1]:
                pred_label.append(1)
            else:
                pred_label.append(0)
    return metrics.accuracy_score(labels, pred_label)


def FOIL_ex(model, mode):
    images, refs, candidates, labels = load_data(mode)
    score_dict = model(images, candidates, refs, reduction=False)

    res_dict = {}
    for score_name, score in score_dict.items():
        acc = cal_acc(score, labels)
        res_dict[score_name] = round(100 * acc, 2)

    return res_dict
