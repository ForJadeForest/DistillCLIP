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
    for pred, label, category in zip(preds, label_list, category_list):
        if pred == label:
            correct[idx2cat[category]] += 1

        total[idx2cat[category]] += 1
    acc = {}
    for k, v in correct.items():
        acc[k] = v / total[k]
    return acc


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
    for i in range(5):
        score1_dict = model(image_path_list, candidate1_list, refs_list[i])
        score2_dict = model(image_path_list, candidate2_list, refs_list[i])
        ref_score_list.append((score1_dict['ref_score'], score2_dict['ref_score']))

    if model.is_clip:
        acc = cal_acc(score1_dict['score'], score2_dict['score'], label_list, category_list)
        res_dict['score'] = postprocess(acc)

    ref_acc = reduction_ref_res(ref_score_list, label_list, category_list)
    res_dict['ref_score'] = postprocess(ref_acc)

    return res_dict
