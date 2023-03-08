import os.path

import pandas as pd
import torch

from test.utils.load_model import load_model
from test.utils.run_all_ex import run_ex


def save_kendall(total_res, save_dir):
    table1_name = ['flickr8k_expert', 'flickr8k_cf', 'composite']
    table1 = {}
    for n in table1_name:
        ex_res = {}
        for model_name, model_res in total_res.items():
            for score_name, score in model_res[n].items():
                ex_res[model_name + '--' + score_name] = score
        table1[n] = ex_res

    table1 = pd.DataFrame(table1)
    table1.to_csv(os.path.join(save_dir, 'kendall.csv'))


def save_pascal(total_res, save_dir):
    table2 = {}
    for model_name, model_res in total_res.items():
        for score_name, score in model_res['pascal50s'].items():
            table2[model_name + '--' + score_name] = score
    table2 = pd.DataFrame(table2).T
    table2.to_csv(os.path.join(save_dir, 'pascal50s.csv'))


def save_FOIL(total_res, save_dir):
    single_ref_res = {}
    multi_ref_res = {}
    for model_name, model_res in total_res.items():
        for score_name, score in model_res['FOIL-1ref'].items():
            single_ref_res[model_name + '--' + score_name] = {'1-ref': score}
        for score_name, score in model_res['FOIL-4ref'].items():
            multi_ref_res[model_name + '--' + score_name] = {'4-ref': score}
    df1 = pd.DataFrame(single_ref_res)
    df2 = pd.DataFrame(multi_ref_res)
    table3 = pd.concat([df1, df2]).T
    table3.to_csv(os.path.join(save_dir, 'FOIL.csv'))


def postprocess(total_res, save_dir, ex_list=None):
    if ex_list is None:
        ex_list = ['kendall', 'pascal50s', 'FOIL']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if 'kendall' in ex_list:
        save_kendall(total_res, save_dir)
    if 'pascal50s' in ex_list:
        save_pascal(total_res, save_dir)
    if 'FOIL' in ex_list:
        save_FOIL(total_res, save_dir)


if __name__ == '__main__':
    model_name_list = [
        # 'WP&Single',
        # 'WP&SR',
        # 'WP+MD&Single',
        # 'L-CLIP',
        # 'CE-CLIP',
        # 'bleu',
        # 'meteor',
        # 'rouge',
        # 'cider',
        # 'spice',
        # 'Teacher',
        'CE-L-CLIP'
    ]
    ex_list = ['kendall', 'pascal50s', 'FOIL']
    with torch.autocast('cuda'):
        with torch.no_grad():
            total_res = {}
            for model_name in model_name_list:
                print(f'Now load the {model_name} model')
                model = load_model(model_name, device='cuda:6', use_PTB=False, use_fp16=True)
                res = run_ex(model, ex_list)
                total_res[model_name] = res
            postprocess(total_res, './CE-L-CLIP_result', ex_list)
