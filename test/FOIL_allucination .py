import collections
import json
# import warnings
import torch
import clip
import clip_score
from sklearn import metrics
import utils

mode = 0  # 1:1 ref 0:4 ref


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


def computeAccOfFOIL(model, device, mode):
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


    '''device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()'''

    with torch.autocast('cuda'):
        image_feats = clip_score.extract_all_images(
            images, model, device, batch_size=1024, num_workers=8)

        # get image-text clipscore
        _, per_instance_image_text, candidate_feats = clip_score.get_clip_score(
            model, image_feats, candidates, device)

        # get text-text clipscore
        _, per_instance_text_text = clip_score.get_refonlyclipscore(
            model, refs, candidate_feats, device)

    # F-score
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (
            per_instance_image_text + per_instance_text_text)


    clip_labels = []
    for num,clip_result in enumerate(per_instance_image_text):
        if (num % 2) == 0:
            if clip_result >= per_instance_image_text[num + 1]:
                clip_labels.append(1)
            else:
                clip_labels.append(0)

    print("clipscore {} accuracy is ".format(mode), metrics.accuracy_score(labels, clip_labels))

    refClip_labels = []
    for num, refclip_result in enumerate(refclipscores):
        if (num % 2) == 0:
            if refclip_result >= per_instance_image_text[num + 1]:
                refClip_labels.append(1)
            else:
                refClip_labels.append(0)

    print("Refclipscore accuracy is ", metrics.accuracy_score(labels, refClip_labels))


    other_metrics = utils.get_all_metrics(refs, candidates, return_per_cap=False)
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

        print("{} accuracy is ".format(k), metrics.accuracy_score(labels, metric_labels))


def main():

    args = utils.get_args()
    device = args.device
    #clip_model = utils.get_model(device, use_fp16=args.fp16)
    clip_model = []
    if args.use_origin:
        print("=" * 10 + "composite Tau-c; Using model: origin" + "=" * 10)
        computeAccOfFOIL(clip_model, device, mode)
    image_path = args.image_path
    text_path = args.text_path
    clip_model = utils.get_model(device, image_path, text_path, use_fp16=args.fp16)
    print("=" * 10 + "composite Tau-c; Using model: distilled" + "=" * 10)
    computeAccOfFOIL(clip_model, device, mode)



if __name__ == '__main__':
    main()