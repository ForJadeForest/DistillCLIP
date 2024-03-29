"""
generate the config structure of an experiment
"""
import argparse
from pathlib import Path


def gene_desc(desc_path):
    content = 'Ex target: \n'
    desc_path.touch()
    with open(desc_path, 'w') as f:
        f.write(content)


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-e', '--ex_name', type=str, help='the experiment name')
    parse.add_argument('-v', '--v_num', type=int, help='the number of the version you need')
    parse.add_argument('-c', '--config', type=str, help='the config path', default='./config')
    parse.add_argument('-t', '--template', type=str, help='the template config path',
                       default='./config/template.yaml')
    return parse.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.config = Path(args.config)
    args.template = Path(args.template)
    if not (args.config / args.ex_name).exists():
        Path.mkdir(args.config / args.ex_name)
    if args.template:
        with open(args.template, 'r', encoding='utf8') as template_f:
            with open(args.config / args.ex_name / 'share.yaml', 'w', encoding='utf8') as out_f:
                out_f.write(template_f.read())
    desc_file = args.config / args.ex_name / 'desc.txt'
    gene_desc(desc_file)

    for i in range(args.v_num):
        ex_path = args.config / args.ex_name / 'version_{}'.format(i)
        version_path = ex_path / 'version.yaml'
        if not ex_path.exists():
            Path.mkdir(ex_path)
        version_path.touch()
        desc_file = ex_path / 'detail_desc.txt'
        desc_file.touch()
