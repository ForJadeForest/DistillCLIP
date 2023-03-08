from random import sample

from test.utils.ex_script import composite_ex, pascal_ex, FOIL_ex, flickr8k_expert_ex, flickr8k_cf_ex
from test.preprocess.pascal50s_ex_dataset import Pascal50sDataset


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
        sample_list = []
        for i in range(5):
            sample_refs = sample(refs, 5)
            sample_list.append(sample_refs)
        refs_list.append(sample_list)
        category_list.append(category)
        label_list.append(label)
    refs_list = list(zip(*refs_list))
    return image_path_list, candidate1_list, candidate2_list, refs_list, label_list, category_list


def run_ex(model, ex_name_list):
    res_dict = {}
    for ex_name in ex_name_list:
        if ex_name == 'kendall':
            composite_root_dir = '/data/ll/composite'
            res_dict['composite'] = composite_ex(model, composite_root_dir)
            print('Now begin flickr8k_expert_ex')
            flickr8k_root_dir = '/data/pyz/data/flickr8k'
            res_dict['flickr8k_expert'] = flickr8k_expert_ex(model, root_dir=flickr8k_root_dir)

            print('Now begin flickr8k_cf_ex')
            flickr8k_root_dir = '/data/pyz/data/flickr8k'
            res_dict['flickr8k_cf'] = flickr8k_cf_ex(model, root_dir=flickr8k_root_dir)
        elif ex_name == 'pascal50s':
            print('Now begin pascal_ex')
            res_dict['pascal50s'] = pascal_ex(model, *prepare_pascal_data())
        elif ex_name == 'FOIL':
            print('Now begin FOIL_ex with 4 ref')
            res_dict['FOIL-4ref'] = FOIL_ex(model, mode=0)

            print('Now begin FOIL_ex with 1 ref')
            res_dict['FOIL-1ref'] = FOIL_ex(model, mode=1)
        else:
            raise ValueError(f'the ex_name error! should in [ "kendall", "pascal50s", "FOIL" ], but got {ex_name}')
    return res_dict


def run_all_ex(model):
    return run_ex(model, ['kendall', 'pascal50s', 'FOIL'])


if __name__ == '__main__':
    from test.utils.load_clip_model.get_model import load_version
    from test.utils.load_clip_model.default_model import CLIPModelLoadConfig
    from test.test_model.n_gram_model import NGramModel
    model_config = CLIPModelLoadConfig(
        clip_path='/home/pyz32/code/Dis_CLIP/CLIPDistillation/34i883eb/checkpoints/last-v1.ckpt'
    )
    model = load_version(model_config, 'L-CLIP', device='cuda:3', use_fp16=True)
    # model = NGramModel('bleu')
    res = run_all_ex(model)
    print(res)
