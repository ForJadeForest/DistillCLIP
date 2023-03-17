from random import sample
import numpy as np

from test.preprocess.pascal50s_ex_dataset import Pascal50sDataset

idx2cat = {1: 'HC', 2: 'HI', 3: 'HM', 4: 'MM'}


def cal_acc(score1, score2, label_list, category_list):
    """
    返回每一个category中的准确率
    :param score1:
    :param score2:
    :param label_list:
    :param category_list:
    :return: A dict {'HC': acc, 'HI': acc, ....}
    """
    keys = idx2cat.values()
    correct, ref_correct, total = ({k: 0 for k in keys} for _ in range(3))
    preds = (score1 < score2).astype('int').tolist()
    return preds


def postprocess(acc):
    for k, v in acc.items():
        acc[k] = round(100 * v, 2)

    acc['mean'] = sum(acc.values()) / len(acc)
    acc['mean'] = round(acc['mean'], 2)
    return acc


def reduction_ref_res(ref_score_list, label_list, category_list):
    ref_multi_ex_res = []
    for ref_score1, ref_score2 in ref_score_list:
        ref_acc = cal_acc(ref_score1, ref_score2, label_list, category_list)
        ref_multi_ex_res.append(ref_acc)

    ref_acc_res = {k: 0 for k in idx2cat.values()}
    for key in idx2cat.values():
        for ref_acc in ref_multi_ex_res:
            ref_acc_res[key] += ref_acc[key]
        ref_acc_res[key] /= 5
    return ref_acc_res


def pascal_ex(model, image_path_list, candidate1_list, candidate2_list, refs_list, label_list, category_list):
    ref_score_list = []
    res_dict = {}

    score1_dict = model(image_path_list, candidate1_list, refs_list)
    score2_dict = model(image_path_list, candidate2_list, refs_list)
    pred = {}
    if model.is_clip:
        preds = cal_acc(score1_dict['score'], score2_dict['score'], label_list, category_list)
        pred['score'] = preds
        res_dict['score_1'] = score1_dict['score']
        res_dict['score_2'] = score2_dict['score']

    preds = cal_acc(score1_dict['ref_score'], score2_dict['ref_score'], label_list, category_list)
    pred['ref_score'] = preds
    res_dict['ref_score_1'] = score1_dict['ref_score']
    res_dict['ref_score_2'] = score2_dict['ref_score']
    return res_dict, pred


from test.utils.load_model import load_model


def prepare_pascal_data():
    dataset = Pascal50sDataset(root='/home/pyz32/code/Dis_CLIP/test/test_dataset/Pascal-50s',
                               voc_path='/data/pyz/data/VOC/VOC2010')
    image_path_list = []
    candidate1_list = []
    candidate2_list = []
    refs_list = []
    category_list = []
    label_list = []
    for data in dataset:
        image, candidate1, candidate2, refs, category, label = data
        image_path_list.append(image)
        candidate1_list.append(candidate1)
        candidate2_list.append(candidate2)

        refs_list.append(sample(refs, 5))

        category_list.append(category)
        label_list.append(label)

    return image_path_list, candidate1_list, candidate2_list, refs_list, label_list, category_list


clip_model = load_model('L-CLIP', device='cuda:2')
cider_model = load_model('cider')
flickr8k_root_dir = '/data/pyz/data/flickr8k'
image_path_list, candidate1_list, candidate2_list, refs_list, label_list, category_list = prepare_pascal_data()
clip_res, clip_pred = pascal_ex(clip_model, image_path_list, candidate1_list, candidate2_list, refs_list, label_list,
                                category_list)
cider_score, cider_pred = pascal_ex(cider_model, image_path_list, candidate1_list, candidate2_list, refs_list,
                                    label_list, category_list)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


res = []
for i, (cider, h_r) in enumerate(zip(cider_pred['ref_score'], label_list)):
    if cider != h_r and clip_pred['score'][i] == h_r and clip_pred['ref_score'][i] == h_r and idx2cat[
        category_list[i]] == 'HC':
        res.append(i)
        print(
            f'cider pred: {cider}, clip pred: {clip_pred["score"][i]}, ref clip pred: {clip_pred["ref_score"][i]}, true label: {h_r}')
        # print(cider, h_r, clip_pred['ref_score'][i], clip_pred['score'][i])

        print(f'candidate0 cider score: {cider_score["ref_score_1"][i]}, clip_score: {clip_res["score_1"][i]}, '
              f'ref_clip_score: {clip_res["ref_score_1"][i]}')
        print(f'candidate1 cider score: {cider_score["ref_score_2"][i]}, clip_score: {clip_res["score_2"][i]}, '
              f'ref_clip_score: {clip_res["ref_score_2"][i]}')

        # print(cider_score['ref_score_1'][i], clip_res['score_1'][i], clip_res['ref_score_1'][i])
        # print(cider_score['ref_score_2'][i], clip_res['score_2'][i], clip_res['ref_score_2'][i])
        print(image_path_list[i])
        print(refs_list[i])
        print('candidate0: ', candidate1_list[i])
        print('candidate1: ', candidate2_list[i])
        print(idx2cat[category_list[i]])
        print('=' * 20)
