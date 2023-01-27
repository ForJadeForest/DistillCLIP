
import json
from pathlib import Path
import pandas as pd
import scipy
import torch.cuda
from utils import get_model
from clip_score import get_clip_score

model2company = {
    'kolarmartin': 'Brno University',
    'karpathy': 'NeuralTalk',
    'rakshithShetty': 'PicSOM',
    # 作者在附录中提到JunhuaMao 和mRNN_Share.JMao的val输出数据是一样的
    # 但是在测试集上的评分有所差异。因此在junhuaMao上也用m-RNN进行结果的计算
    'junhuaMao': 'm-RNN',
    # 'junhuaMao': 'm-RNN (Baidu/ UCLA)',
    'OriolVinyals': 'Google',
    'myamaguchi': 'MIL',
    'Q.Wu': 'ACVT',
    'jeffdonahue': 'Berkeley LRCN',
    'mRNN_Share.JMao': 'm-RNN',
    'TsinghuaBigeye': 'Tsinghua Bigeye',
    'ryank': 'MLBL',
    'kelvin_xu': 'Montreal/Toronto'
}


model_list = model2company.keys()


def cal_metric(model_name, clip_model, images_root_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(images_root_dir, Path):
        images_root_dir = Path(images_root_dir)
    root_dir = Path('coco_captioning_challenge') / model_name
    data_path = root_dir.glob('captions_val2014*_results.json')[0]
    with open(data_path, 'r')as f:
        data = json.load(f)
    id2text = {
        d['image_id']: d['caption'] for d in data
    }
    image_id_list = id2text.keys()
    images_path = [images_root_dir / image_id for image_id in image_id_list]
    text = [id2text[k] for k in image_id_list]

    return get_clip_score(clip_model, images_path, text, device)


def main():
    clip_model = get_model()
    images_root_dir = ''
    human_metric = pd.read_csv('./coco_captioning_challenge/leaderboard.csv').dropna(axis=1)
    human_metric = human_metric.set_index('Unnamed: 1').T
    clip_score_res = []
    human_metric_res_m1 = []
    human_metric_res_m2 = []
    for model_name in model_list:
        mean_score, per_score, _ = cal_metric(model_name, clip_model, images_root_dir)
        clip_score_res.append(mean_score)
        human_metric_res_m1.append(human_metric[model2company[model_name]]['M1'])
        human_metric_res_m2.append(human_metric[model2company[model_name]]['M2'])

    print('CLIPScore Spearmanr: {}'.format(scipy.stats.spearmanr(clip_score_res, human_metric_res_m1)[0]))
    print('CLIPScore Spearmanr: {}'.format(scipy.stats.spearmanr(clip_score_res, human_metric_res_m2)[0]))







