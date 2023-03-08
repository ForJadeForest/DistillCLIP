from test.test_model.n_gram_model import NGramModel

from test.utils.load_clip_model.get_model import load_version
from test.utils.load_clip_model.default_model import model_name_map

n_gram_model_name = ['bleu', 'meteor', 'rouge', 'cider', 'spice']


def load_model(model_name: str, device=None, use_PTB=False, use_fp16=True, model_config=None):
    if model_name.lower() in n_gram_model_name:
        print(f'load the {model_name.lower()} n-gram-based model')
        model = NGramModel(model_name, use_PTB)
    else:
        if model_config is not None:
            return load_version(model_config, model_name, device, use_fp16)
        print(f'load the default config of {model_name}')
        model_config = model_name_map[model_name]
        model = load_version(model_config, model_name, device, use_fp16)
    return model
