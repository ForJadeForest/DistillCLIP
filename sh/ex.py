import yaml
from pathlib import Path
import argparse


def get_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('-a', '--all', action='store_true', help='whether to deal with all experiment and the versions')
    parse.add_argument('-n', '--name', type=str, help='the experiment name')
    parse.add_argument('-v', '--version', type=str, help='th experiment\'s version, if is none, deal all versions')
    parse.add_argument('-c', '--config', type=str, help='the config path', default='../config')
    return parse.parse_args()


def generate_config(ex_name, version_name, config_path):
    share_para_path = config_path / ex_name / 'share.yaml'
    with open(share_para_path, 'r', encoding='utf8') as f:
        share_para = yaml.load(f, yaml.FullLoader)
    version_para_path = config_path / ex_name / version_name / 'version.yaml'
    with open(version_para_path, 'r', encoding='utf8') as f:
        version_para = yaml.load(f, yaml.FullLoader)
    para = {}
    para.update(share_para)
    for k in para:

        if version_para and k in version_para:
            para[k].update(version_para[k])
    return para, config_path / ex_name / version_name


if __name__ == '__main__':
    args = get_args()
    args.config = Path(args.config)
    if not args.all:
        para, save_path = generate_config(args.name, args.version, args.config)
        with open(save_path / 'final.yaml', 'w', encoding='utf8')as f:
            f.write(yaml.dump(para))
    else:
        for ex in args.config.iterdir():
            if not ex.is_dir():
                continue
            for v in [d for d in ex.iterdir() if d.is_dir()]:
                para, save_path = generate_config(ex.name, v.name, args.config)
                with open(save_path / 'final.yaml', 'w', encoding='utf8')as f:
                    f.write(yaml.dump(para))
