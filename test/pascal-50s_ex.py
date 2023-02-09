from random import sample

import numpy as np
import torch
from test.preprocess.pascal50s_ex_dataset import Pascal50sDataset
from test.clip_score import get_clip_score, get_ref_clip_score
from test.utils import get_model, get_args

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

    for k, v in correct.items():
        print(f'the result {k} acc: {v / total[k]}')
    return acc


def logout(acc_list):
    print('=' * 80)
    res = {k: 0 for k in idx2cat.values()}
    for key in idx2cat.values():
        for acc in acc_list:
            res[key] += acc[key]
        res[key] /= repeat_times

    print('the final result: ')

    for k, v in res.items():
        print(f'the result {k} acc: {v}')

    print(f'the mean value of the acc is {sum(res.values()) / len(res.values())}')
    print('=' * 80)


def cal_clip_score(pascal_dataset, model):
    image_path_list = []
    candidate1_list = []
    candidate2_list = []
    refs_list = []
    category_list = []
    label_list = []
    for data in pascal_dataset:
        image, candidate1, candidate2, refs, category, label = data
        image_path_list.append(image)
        candidate1_list.append(candidate1)
        candidate2_list.append(candidate2)
        refs_list.append(sample(refs, 5))
        category_list.append(category)
        label_list.append(label)
    with torch.autocast('cuda'):
        score1 = get_clip_score(model, image_path_list, candidate1_list, device)[1]
        score2 = get_clip_score(model, image_path_list, candidate2_list, device)[1]
        ref_score1 = np.array(get_ref_clip_score(model, image_path_list, refs_list, candidate1_list, device)[1])
        ref_score2 = np.array(get_ref_clip_score(model, image_path_list, refs_list, candidate2_list, device)[1])

    acc = cal_acc(score1, score2, label_list, category_list)
    ref_acc = cal_acc(ref_score1, ref_score2, label_list, category_list)
    return acc, ref_acc


def cal_one_model_res(clip_model, repeat_times):
    acc_list = []
    ref_acc_list = []
    for i in range(repeat_times):
        dataset = Pascal50sDataset(root='test_dataset/Pascal-50s', voc_path='/data/pyz/data/VOC/VOC2010')
        acc, ref_acc = cal_clip_score(dataset, clip_model)
        acc_list.append(acc)
        ref_acc_list.append(ref_acc)

    print('The no ref acc result')
    logout(acc_list)
    print('The ref acc result')
    logout(ref_acc_list)


if __name__ == '__main__':
    args = get_args()
    device = args.device
    repeat_times = 5

    print('=' * 10 + 'begin distillation model pascal-50s ex!' + '=' * 10)
    image_path = args.image_path
    text_path = args.text_path
    model = get_model(device, image_path, text_path, use_fp16=args.fp16)
    cal_one_model_res(model, repeat_times)

    print('=' * 10 + 'begin original model pascal-50s ex!' + '=' * 10)
    model = get_model(device, use_fp16=args.fp16)
    cal_one_model_res(model, repeat_times)
