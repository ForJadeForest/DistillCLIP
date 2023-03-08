import torch

from test.utils.load_model import load_model
from test.utils.run_all_ex import run_all_ex
from test.run_all_model import postprocess
from test.utils.load_clip_model.get_model import get_all_version_path


model_name_list = ['WP&Single', 'WP&SR', 'WP+MD&Single', 'L-CLIP', 'CE-CLIP', 'CE-L-CLIP']
model_name_list = ['CE-CLIP']
root_path = '/data/share/pyz/Dis_CLIP/final'
total_res = {}
with torch.autocast('cuda'):
    with torch.no_grad():
        for model_name in model_name_list:

            all_version_config = get_all_version_path(root_path, model_name)
            print(f'[INFO] ==> Begin {model_name}, the model number is {len(all_version_config)}')
            for version_config_name, config in all_version_config.items():
                model = load_model(version_config_name, device='cuda', model_config=config)
                one_model_res = run_all_ex(model)
                total_res[version_config_name] = one_model_res
                postprocess(total_res, './grid_search_ce_clip_result')