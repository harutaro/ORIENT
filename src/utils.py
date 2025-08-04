import os
import torch
import random
import numpy as np
from argparse import ArgumentParser

import approach
from datasets.dataset_config import dataset_config
from networks import allmodels

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)


def get_exec_args():
    # Arguments
    parser = ArgumentParser(
        description='FACIL - Framework for Analysis of Class Incremental Learning'
    )

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0, help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='出力先のディレクトリ (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str, help='実験名 (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='Seed値 (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='ロガーの種類(disk, tensorboard) (default=%(default)s)',
                        nargs='*', metavar='LOGGER')
    parser.add_argument('--save-models', action='store_true',
                        help='モデルの保存を行う (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Headのグラフを出力する (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='CUDNN deterministicを無効にする (default=%(default)s)')

    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='使用するデータセット指定 (default=%(default)s)',
                        nargs='+', metavar='DATASET')
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='データローダのサブプロセス数 (default=%(default)s)')
    parser.add_argument('--pin-memory', action='store_true', required=False,
                        help='https://pytorch.org/docs/stable/data.html#memory-pinning参照。(default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='バッチサイズ (default=%(default)s)')
    parser.add_argument('--num-tasks', default=5, type=int, required=False, help='タスク数 (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='第1タスクのクラス数 (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='該当のタスクまでで実行停止 (default=%(default)s)')
    # dataset args (Extra Arguments Added by KDDI)
    parser.add_argument('--validation', default=0.1, type=float, required=False,
                        help='学習用データの内、Validationデータの割合 (default=%(default))')

    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='使用するモデル (default=%(default)s)', metavar='NETWORK')
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Headを削除しない (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='事前学習済みパラメータ使用 (default=%(default)s)')

    # grid search args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='GridSearchを行うタスク数 (-1,0: 無効) (default=%(default)s)')
    parser.add_argument('--lr-first', default=None, type=float, required=False,
                        help='第1タスクのlrを指定 (default=%(default)s)')

    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='継続学習の手法 (default=%(default)s)', metavar='APPROACH')
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='最大Epoch数 (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False, help='学習率 (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='学習を打ち切るlrの値 (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='学習減衰比 lr-patienceの条件を満たすと1/lr_factorする(default=%(default)s)')
    parser.add_argument('--lr-patience', default=10, type=int, required=False,
                        help='指定のEpoch数分 validation lossの最低値更新がなかった場合に学習率減衰(default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='勾配Clipping (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='SGDのmomentum (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='重み減衰 (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='ウォームアップ Epoch数(default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='ウォームアップの学習率減衰比 (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    return parser.parse_known_args()
