import argparse
from pathlib import Path
import os

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-e', '--ex_name', type=str, help='the experiment name')
    parse.add_argument('-v', '--v_num', type=str, help='the number of the version')
    parse.add_argument('-c', '--config', type=str, help='the config path', default='./config')
    parse.add_argument('-b', '--begin_ver', type=int, default=None)
    parse.add_argument('-t', '--end_ver', type=int, default=None)
    parse.add_argument('--all_ver', action='store_true', help='whether to deal with all experiment and the versions')
    parse.add_argument('--all_ex', action='store_true', help='whether to deal with all experiment and the ex')
    parse.add_argument('-n', '--n_ver', nargs='+', help='the version you want to run, eg. -n 3 4 8')
    parse.add_argument('-o', '--other_para', type=str, help='Other parameter for yaml file')
    return parse.parse_args()


def run_version(ex_name, ver_num, config_path):
    ex_path = config_path / ex_name
    share_config_path = ex_path / 'share.yaml'
    version_config_path = ex_path / ver_num / 'version.yaml'
    config_path = [str(share_config_path), str(version_config_path)]
    print('=' * 33 + 'Now is Running [{}] experiment and [{}]'.format(ex_name, ver_num) + '=' * 33)
    command = 'python ./main.py fit -c ' + ' -c '.join(config_path)
    if args.other_para:
        command += ' '
        command += args.other_para
    # print('\nrun command: ' + command + '\n')
    os.system('python ./main.py fit -c ' + ' -c '.join(config_path))
    print('=' * 34 + '[{}] experiment and [{}] is done!'.format(ex_name, ver_num) + '=' * 34)
    print('\n' * 3)


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
    elif args.ex_name and (args.begin_ver is not None or args.end_ver is not None):
        # 完成指定实验的部分版本
        ex_path = args.config / args.ex_name
        ver_file = [file for file in sorted(ex_path.iterdir()) if file.is_dir()]
        if args.begin_ver is None:
            args.begin_ver = 0
        if args.end_ver is None or args.end_ver == -1:
            args.end_ver = len(ver_file)
        assert args.begin_ver <= len(ver_file) and len(ver_file) >= args.end_ver, \
            f'the begin_ver or end_ver must be smaller than {len(ver_file)}, but got {args.begin_ver, args.end_ver}'
        for v in [file for file in sorted(ex_path.iterdir()) if file.is_dir()][args.begin_ver: args.end_ver]:
            run_version(ex_path.name, v.name, args.config)
    elif args.ex_name and args.n_ver:
        ex_path = args.config / args.ex_name
        for n in args.n_ver:
            ver_file = [file for file in sorted(ex_path.iterdir()) if file.is_dir()]
            if 0 <= int(n) < len(ver_file):
                run_version(args.ex_name, f'version_{n}', args.config)
            else:
                print(f'the number of {n} is invalid, the num should in [0, {len(ver_file)})')
