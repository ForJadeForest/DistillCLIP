import argparse
from pathlib import Path

import yaml

"""
生成不同目的的模板
1. 训练模板 fit
2. 找瓶颈模板 fit 
3. lr finder 模板 tune 
4. 
"""


def gene_trainer_para(target):
    para = {}
    if target == 't':
        para = {'trainer': {
            'accumulate_grad_batches': 1,
            'accelerator': 'gpu',
            'max_epochs': 50,
            'min_epochs': 3,
            'max_steps': -1,
            'devices': [0, 1],
            'precision': 32,
            'log_every_n_steps': 100,
            'strategy': 'ddp',
            'logger': {
                'class_path': 'pytorch_lightning.loggers.tensorboard.TensorBoardLogger',
                'init_args': {
                    'save_dir': '/data/pyz/res',
                    'name': 'TextEncoder',
                    'version': None,
                }
            },
            'enable_progress_bar': True,
            'check_val_every_n_epoch': 1,
            #  callbacks:
            'callbacks': [
                {
                    'class_path': 'LearningRateMonitor'
                },
                {
                    'class_path': 'EarlyStopping',
                    'init_args': {
                        'monitor': 'val/loss'
                    }
                },
                {
                    'class_path': 'ModelCheckpoint',
                    'init_args': {
                        'filename': '{epoch}-val_acc{hp_metric/stu_acc_top1:.2f}-loss{val/loss}',
                        'monitor': 'hp_metric/stu_acc_top1',
                        'save_last': True,
                        'save_top_k': 2,
                        'mode': 'max',
                        'auto_insert_metric_name': False,
                    }
                },
                {
                    'class_path': 'ModelSummary',
                    'init_args': {
                        'max_depth': 2,
                    }
                }
            ], }
        }
    elif target.startswith('b'):
        if target == 'bs':
            para = {
                'trainer': {
                    'profiler': {
                        'class_path': 'pytorch_lightning.profiler.SimpleProfiler',
                        'init_args': {
                            'dirpath': './',
                            'filename': 'profile.log'
                        }

                    },
                    'fast_dev_run': True,
                    'accelerator': 'gpu',
                    'devices': 1,
                }
            }
        elif target == 'ba':
            para = {
                'trainer': {
                    'profiler': {
                        'class_path': 'pytorch_lightning.profiler.AdvancedProfiler',
                        'init_args': {
                            'dirpath': './',
                            'filename': 'profile.log'
                        }

                    },
                    'fast_dev_run': True,
                    'accelerator': 'gpu',
                    'devices': 1,
                }
            }
    elif target == 'l':
        para = {
            'trainer': {
                'auto_lr_find': True,
                'accelerator': 'gpu',
                'devices': 1,
            }
        }
    else:
        raise ValueError('the target error, should in [t, bs, ba, l]')

    return para


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-t', '--target', type=str, help='the target of the template name', default='t')
    parse.add_argument('-c', '--config', type=str, help='the config path', default='./config')
    parse.add_argument('-n', '--name', type=str, help='the name of the template file', default='template.yaml')
    return parse.parse_args()


def gene_yaml_file(args, para):
    if args.target != 't':
        test_fold = args.config / 'test'
        test_fold.mkdir()

        with open(test_fold / (args.target + '_' + args.name), 'w') as f:
            yaml.dump(para, f)
    else:
        with open(args.config / args.name, 'w') as f:
            yaml.dump(para, f)


if __name__ == '__main__':
    args = get_args()
    args.config = Path(args.config)
    trainer_para = gene_trainer_para(args.target)
    gene_yaml_file(args, trainer_para)
