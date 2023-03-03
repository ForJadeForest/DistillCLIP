from random import sample

import numpy as np
import torch
from test.preprocess.pascal50s_ex_dataset import Pascal50sDataset
from test.clip_score import get_clip_score, get_ref_clip_score, get_all_clip_score
from test.utils import get_model, get_args, total_ex

idx2cat = {1: 'HC', 2: 'HI', 3: 'HM', 4: 'MM'}


def cal_acc(score1, score2, label_list, category_list):
    keys = idx2cat.values()
    correct, ref_correct, total = ({k: 0 for k in keys} for _ in range(3))
    preds = (score1 < score2).astype('int').tolist()
    for pred, label, category in zip(preds, label_list, category_list):
        if pred == label:
            correct[idx2cat[category]] += 1

        total[idx2cat[category]] += 1
    acc = {}
    for k, v in correct.items():
        acc[k] = v / total[k]
    return acc


def logout(res):
    print('the final result: ')
    final_res = {}
    for k, v in res.items():
        final_res[k] = round(100 * v, 2)
        print(f'the result {k} acc: {final_res[k]}')

    mean_score = sum(res.values()) / len(res.values())
    final_res['mean'] = round(100 * mean_score, 2)
    print(f'the mean value of the acc is {mean_score}')
    return final_res


def cal_pascal_acc(model, device,
                   image_path_list, candidate1_list, candidate2_list,
                   refs_list, label_list, category_list):
    with torch.autocast('cuda'):
        ref_score_list = []
        for i in range(5):
            _, score1, _, ref_score1, = get_all_clip_score(model, image_path_list, refs_list[i], candidate1_list, device)
            _, score2, _, ref_score2, = get_all_clip_score(model, image_path_list, refs_list[i], candidate2_list, device)
            ref_score_list.append((ref_score1, ref_score2))
    acc = cal_acc(score1, score2, label_list, category_list)

    ref_multi_ex_res = []
    for ref_s1, ref_s2 in ref_score_list:
        ref_acc = cal_acc(ref_s1, ref_s2, label_list, category_list)
        ref_multi_ex_res.append(ref_acc)

    ref_acc_res = {k: 0 for k in idx2cat.values()}
    for key in idx2cat.values():
        for ref_acc in ref_multi_ex_res:
            ref_acc_res[key] += ref_acc[key]
        ref_acc_res[key] /= 5

    return acc, ref_acc_res


def cal_one_model_res(clip_model, device, *data_args):

    acc, ref_acc = cal_pascal_acc(clip_model, device, *data_args)

    final_res = {}
    print('The no ref acc result')
    final_res['Pascal-50s CLIP-S'] = logout(acc)
    print('The ref acc result')
    final_res['Pascal-50s Ref-CLIP-S'] = logout(ref_acc)
    return final_res


def main(args, *data_args):
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    model = get_model(device, load_teacher, clip_path, image_path, text_path)

    return cal_one_model_res(model, device, *data_args)


if __name__ == '__main__':
    args = get_args()
    dataset = Pascal50sDataset(root='test_dataset/Pascal-50s', voc_path='/data/pyz/data/VOC/VOC2010')
    image_path_list = []
    candidate1_list = []
    candidate2_list = []
    refs_list = []
    repeat_refs_list = []
    category_list = []
    label_list = []
    for data in dataset:
        image, candidate1, candidate2, refs, category, label = data
        image_path_list.append(image)
        candidate1_list.append(candidate1)
        candidate2_list.append(candidate2)
        temp = []
        for i in range(5):
            sample_refs = sample(refs, 5)
            temp.append(sample_refs)
        refs_list.append(temp)
        category_list.append(category)
        label_list.append(label)
    refs_list = list(zip(*refs_list))
    total_ex(args, main, image_path_list, candidate1_list, candidate2_list,
             refs_list, label_list, category_list)
