import json
from pathlib import Path
import pandas as pd
from scipy import stats
import torch.cuda
from utils import get_model, get_args, total_ex, get_all_metrics
from clip_score import extract_all_images, get_all_clip_score

model2company = {
    'kolarmartin': 'Brno University',
    'karpathy': 'NeuralTalk',
    'rakshithShetty': 'PicSOM',
    # 作者在附录中提到JunhuaMao 和mRNN_Share.JMao的val输出数据是一样的
    # 但是在测试集上的评分有所差异。因此在junhuaMao上也用m-RNN (Baidu/ UCLA)进行结果的计算
    # 'junhua.mao': 'm-RNN',
    'junhua.mao': 'm-RNN (Baidu/ UCLA)',
    'OriolVinyals': 'Google',
    'myamaguchi': 'MIL',
    'Q.Wu': 'ACVT',
    'jeffdonahue': 'Berkeley LRCN',
    # 'mRNN_share.JMao': 'm-RNN',
    'mRNN_share.JMao': 'm-RNN (Baidu/ UCLA)',
    'TsinghuaBigeye': 'Tsinghua Bigeye',
    'ryank': 'MLBL',
    'kelvin_xu': 'Montreal/Toronto'
}

model_list = model2company.keys()


def filename2id(filename):
    import re
    pattern = re.compile('COCO_val2014_0*(.*).jpg')
    return int(re.findall(pattern, str(filename))[0])


def init_image_features(clip_model, images_root_dir, device):
    images_id_list = sorted(list(images_root_dir.iterdir()))
    images_path = [images_root_dir / p for p in images_id_list]
    with torch.autocast('cuda' if 'cuda' in device else 'cpu'):
        image_features = extract_all_images(images_path, clip_model, device=device)
    return images_id_list, image_features


def cal_metric(model_name, clip_model, images_filename, device, image_features):
    root_dir = Path('../../test_dataset/coco_captioning_challenge') / model_name
    data_path = list(root_dir.glob('captions_val2014*_results.json'))[0]
    with open(data_path, 'r') as f:
        data = json.load(f)
    with open('/data/pyz/data/mscoco/annotations/captions_val2014.json', 'r') as f:
        ref_data = json.load(f)['annotations']
    id2ref = {}
    for d in ref_data:
        if d['image_id'] in id2ref:
            id2ref[d['image_id']].append(d['caption'])
        else:
            id2ref[d['image_id']] = [d['caption']]
    id2text = {
        d['image_id']: d['caption'] for d in data
    }

    text = [id2text[filename2id(k)] for k in images_filename]
    ref_text = [id2ref[filename2id(k)] for k in images_filename]

    with torch.autocast('cuda' if 'cuda' in device else 'cpu'):
        res, _, ref_res, _ = get_all_clip_score(clip_model, image_features, ref_text, text, device)
    other_metrics = get_all_metrics(ref_text, text)
    return res, ref_res, other_metrics


def cal_coco_ex(model, device):
    final_res = {}
    images_root_dir = Path(r'/data/pyz/data/mscoco/val2014')
    images_filename, images_features = init_image_features(model, images_root_dir, device)

    human_metric = pd.read_csv('../../test_dataset/coco_captioning_challenge/leaderboard.csv').dropna(axis=1)
    human_metric = human_metric.set_index('Unnamed: 1').T
    clip_score_res = []
    ref_clip_score_res = []
    human_metric_res_m1 = []
    human_metric_res_m2 = []
    bleu_score = []
    meteor_score = []
    cider_score = []
    spice_score = []
    rouge_score = []
    for model_name in model_list:
        mean_score, ref_mean_score, other_metric = cal_metric(model_name, model, images_filename, device, images_features)
        clip_score_res.append(mean_score)
        ref_clip_score_res.append(ref_mean_score)
        human_metric_res_m1.append(human_metric[model2company[model_name]]['M1'])
        human_metric_res_m2.append(human_metric[model2company[model_name]]['M2'])
        bleu_score.append(other_metric['bleu'][-1])
        meteor_score.append(other_metric['meteor'])
        cider_score.append(other_metric['cider'])
        spice_score.append(other_metric['spice'])
        rouge_score.append(other_metric['rouge'])

    bleu_score_res = {}
    meteor_score_res = {}
    cider_score_res = {}
    spice_score_res = {}
    rouge_score_res = {}
    clip_s_res = {}
    ref_clip_s_res = {}

    m1_spearmanr, m1_p_value = stats.spearmanr(bleu_score, human_metric_res_m1)
    bleu_score_res['M1 Spearman'] = round(100 * m1_spearmanr, 2)
    m2_spearmanr, m1_p_value = stats.spearmanr(bleu_score, human_metric_res_m2)
    bleu_score_res['M2 Spearman'] = round(100 * m2_spearmanr, 2)


    m1_spearmanr, m1_p_value = stats.spearmanr(meteor_score, human_metric_res_m1)
    meteor_score_res['M1 Spearman'] = round(100 * m1_spearmanr, 2)
    m2_spearmanr, m1_p_value = stats.spearmanr(meteor_score, human_metric_res_m2)
    meteor_score_res['M2 Spearman'] = round(100 * m2_spearmanr, 2)


    m1_spearmanr, m1_p_value = stats.spearmanr(cider_score, human_metric_res_m1)
    cider_score_res['M1 Spearman'] = round(100 * m1_spearmanr, 2)
    m2_spearmanr, m1_p_value = stats.spearmanr(cider_score, human_metric_res_m2)
    cider_score_res['M2 Spearman'] = round(100 * m2_spearmanr, 2)


    m1_spearmanr, m1_p_value = stats.spearmanr(spice_score, human_metric_res_m1)
    spice_score_res['M1 Spearman'] = round(100 * m1_spearmanr, 2)
    m2_spearmanr, m1_p_value = stats.spearmanr(spice_score, human_metric_res_m2)
    spice_score_res['M2 Spearman'] = round(100 * m2_spearmanr, 2)

    m1_spearmanr, m1_p_value = stats.spearmanr(rouge_score, human_metric_res_m1)
    rouge_score_res['M1 Spearman'] = round(100 * m1_spearmanr, 2)
    m2_spearmanr, m1_p_value = stats.spearmanr(rouge_score, human_metric_res_m2)
    rouge_score_res['M2 Spearman'] = round(100 * m2_spearmanr, 2)



    m1_spearmanr, m1_p_value = stats.spearmanr(clip_score_res, human_metric_res_m1)
    print(f'CLIPScore for M1 Spearmanr: {m1_spearmanr}, p-value: {m1_p_value}')
    clip_s_res['M1 Spearman'] = round(100 * m1_spearmanr, 2)

    m2_spearmanr, m2_p_value = stats.spearmanr(clip_score_res, human_metric_res_m2)
    print(f'CLIPScore for M2 Spearmanr: {m2_spearmanr}, p-value: {m2_p_value}')
    clip_s_res['M2 Spearman'] = round(100 * m2_spearmanr, 2)

    ref_m1_spearmanr, m1_p_value = stats.spearmanr(ref_clip_score_res, human_metric_res_m1)
    print(f'Ref CLIPScore for M1 Spearmanr: {ref_m1_spearmanr}, p-value: {m1_p_value}')
    ref_clip_s_res['M1 Spearman'] = round(100 * ref_m1_spearmanr, 2)

    ref_m2_spearmanr, m2_p_value = stats.spearmanr(ref_clip_score_res, human_metric_res_m2)
    print(f'Ref CLIPScore for M2 Spearmanr: {ref_m2_spearmanr}, p-value: {m2_p_value}')
    ref_clip_s_res['M2 Spearman'] = round(100 * ref_m2_spearmanr, 2)

    m1_pearsonr, m1_p_value = stats.pearsonr(clip_score_res, human_metric_res_m1)
    print(f'CLIPScore for M1 pearsonr: {m1_pearsonr}, p-value: {m1_p_value}')
    clip_s_res['M1 Pearson'] = round(100 * m1_pearsonr, 2)

    m2_pearsonr, m2_p_value = stats.pearsonr(clip_score_res, human_metric_res_m2)
    print(f'CLIPScore for M2 pearsonr: {m2_pearsonr}, p-value: {m2_p_value}')
    clip_s_res['M2 Pearson'] = round(100 * m2_pearsonr, 2)

    ref_m1_pearsonr, m1_p_value = stats.pearsonr(ref_clip_score_res, human_metric_res_m1)
    print(f'Ref CLIPScore for M1 pearsonr: {ref_m1_pearsonr}, p-value: {m1_p_value}')
    ref_clip_s_res['M1 Pearson'] = round(100 * ref_m1_pearsonr, 2)

    ref_m2_pearsonr, m2_p_value = stats.pearsonr(ref_clip_score_res, human_metric_res_m2)
    print(f'Ref CLIPScore for M2 pearsonr: {ref_m2_pearsonr}, p-value: {m2_p_value}')
    ref_clip_s_res['M2 Pearson'] = round(100 * ref_m2_pearsonr, 2)

    final_res['CLIP-S'] = clip_s_res
    final_res['Ref-CLIP-S'] = ref_clip_s_res

    final_res['bleu'] = bleu_score_res
    final_res['meteor'] = meteor_score_res
    final_res['cider'] = cider_score_res
    final_res['spice'] = spice_score_res
    final_res['rouge'] = rouge_score_res

    return final_res


def main(args):
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    clip_model = get_model(device, args.load_teacher, clip_path, image_path, text_path, args.fp16)
    return cal_coco_ex(clip_model, device)


if __name__ == '__main__':
    args = get_args()
    total_ex(args, main)