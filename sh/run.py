import argparse
import os
from pathlib import Path

# TODO 加入一些描述性话语，比如执行全部命令
# TODO 加入指定部分版本的功能比如指定1，3，4三个version

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-e', '--ex_name', type=str, help='the experiment name')
    parse.add_argument('-v', '--v_num', type=str, help='the number of the version')
    parse.add_argument('-c', '--config', type=str, help='the config path', default='./config')
    parse.add_argument('-b', '--begin_ver', type=int, default=None)
    parse.add_argument('-t', '--end_ver', type=int, default=None)
    parse.add_argument('--all_ver', action='store_true', help='whether to deal with all experiment and the versions')
    parse.add_argument('--all_ex', action='store_true', help='whether to deal with all experiment and the versions')
    return parse.parse_args()


def run_version(ex_name, ver_num, config_path):
    ex_path = config_path / ex_name
    share_config_path = ex_path / 'share.yaml'
    version_config_path = ex_path / ver_num / 'version.yaml'
    config_path = [str(share_config_path), str(version_config_path)]
    print('=' * 20 + 'Now is Running {} experiment and version {}'.format(ex_name, ver_num) + '=' * 20)
    print('run command: python ./main.py fit -c ' + ' -c '.join(config_path))
    os.system('python ./main.py fit -c ' + ' -c '.join(config_path))
    print('=' * 20 + '{} experiment and version {} is done!'.format(ex_name, ver_num) + '=' * 20)


if __name__ == '__main__':
    args = get_args()
    args.config = Path(args.config)

    if args.all_ex:
        # 完成所有的实验
        ex_file = [file for file in sorted(args.config.iterdir()) if file.is_dir()]
        for ex_path in ex_file:
            ver_file = [file for file in sorted(ex_path.iterdir()) if file.is_dir()]
            for v in ver_file:
                run_version(ex_path.name, v.name, args.config)
    elif args.all_ver and args.ex_name is not None:
        # 完成一个实验的所有版本
        ex_path = args.config / args.ex_name
        ver_file = [file for file in sorted(ex_path.iterdir()) if file.is_dir()]
        for v in ver_file:
            run_version(args.ex_name, v.name, args.config)
    elif args.ex_name and args.v_num:
        # 完成指定的一个实验和一个版本
        run_version(args.ex_name, 'version_' + args.v_num, args.config)
    elif args.ex_name and args.begin_ver and args.end_ver:
        # 完成指定实验的部分版本
        ex_path = args.config / args.ex_name
        for v in [file for file in sorted(ex_path.iterdir()) if file.is_dir()][args.begin_ver: args.end_ver]:
            run_version(ex_path.name, v.name, args.config)
