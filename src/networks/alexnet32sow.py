from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

## for SOW class
#from .sow_libs_V0 import SOW_V0
#from .sow_libs_V1 import SOW_V1
#from .sow_libs_V2 import SOW_V2
from .sow_libs_V3 import SOW_V3


__all__ = ("alexnet_32sow")

class AlexNet_32SOW(nn.Module):
    """ 入力：3ch × 32 画素 × 32 画素 """
    """ アーキテクチャ： ＳＯＷ版ＡｌｅｘＮｅｔ """

    def __init__(self, device='cuda', dropout=0.5,
                 fix_features=False, load_features=False, pretrained_path='',
                 num_tasks=10, **kwargs):
        super().__init__()

        ## 初期データ保管ファイルからSOW用の S_max, init_a, ptbl, ranks, ttbl を読み込み
        M = 12
        N = 4096
        fname = '../src/networks/INIT/%04d.dict' % (N)
        print(f'{fname}: loading..')
        sow_dict = torch.load(fname)
        S_max = sow_dict['S_max'].to(device=device)
        init_a = sow_dict['a'].to(device=device)
        ptbl = sow_dict['ptbl'].to(device=device)
        ranks = sow_dict['ranks'].to(device=device)
        ttbl = sow_dict['ttbl'].to(device=device)
        for i in range(1, N) : init_a[i] = 0.0
        # 読み込み結果の確認
        print('  S_max:', S_max)
        print('  init_a:', init_a.shape, init_a.dtype, init_a.device)
        print('  ptbl:', ptbl.shape, ptbl.dtype, ptbl.device)
        #print_ptbl(ptbl)
        #print_pidx(ptbl)
        print('  ranks:', ranks)
        print('  ttbl:', ttbl.shape, ttbl.dtype, ttbl.device)

        # パラメータdropoutが、単数の場合と複数の場合の両方に対応する
        if type(dropout) is list:
            d1 = dropout[0]
            d2 = dropout[1]
        elif type(dropout) is float:
            d1 = dropout
            d2 = dropout
        else:
            assert False

        self.fix_features = fix_features
        self.fix_classifier = False
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 64,23,23
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64,16,16
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # 192,16,16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 192,8,8
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  # 384,8,8
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 256,8,8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 256,8,8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 256,4,4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=d1),
            SOW_V3(M, S_max, init_a, ptbl, ranks, ttbl, num_tasks=num_tasks, device=device, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(p=d2),
            SOW_V2(4096, 4096, num_tasks=num_tasks, GS_rank=GS_rank, device=device, dtype=torch.float64),
            nn.ReLU()
        )
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(4096, num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = "fc"

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location=device)
            # HACK: state_dictからfeatures,classifierを取り出すようにし、
            # model.load_state_dict()に取り出したものを適用するという感じにしたい
            if load_features:
                self.features.load_state_dict(state_dict)
                print(F'features is loaded ({pretrained_path})')
        if fix_features: self.Fix_Features()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float64)
        x = self.classifier(x)
        x = x.to(dtype=torch.float)
        x = self.fc(x)
        return x

    def extra_repr(self) -> str:
        return 'in_channels={}, in_H={}, in_W={}, training={}, fix_features={}, fix_classifier={}'.format(
            3, 32, 32, self.training, self.fix_features, self.fix_classifier
        )

    def train(self, mode):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # For Features Extractor
        if self.fix_features : self.features.eval()
        else: self.features.train(mode)
        # For Main Classifier except Final Classifier
        if self.fix_classifier : self.classifier.eval()
        else: self.classifier.train(mode)
        # For Final Classifier
        self.fc.train(mode)
        return self

    def Fix_Features(self):
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False 
        self.fix_features = True
        print("Fix_Features: Done")
        #print(F'{self.features} --> freezed')

    def Fix_Classifier(self):
        self.classifier.eval()
        for name, param in self.classifier.named_parameters():
            param.requires_grad = False
        self.fix_classifier = True
        print(F'Fix_Classifier: Done (training={self.classifier.training}, fix_classifier={self.fix_classifier})')
        #print(F'{self.classifier} --> freezed')

    def Free_Classifier(self):
        self.classifier.train()
        for name, param in self.classifier.named_parameters():
            if name.endswith('.best_low_rank_s') :
                continue # for s_matrix & fix_matrix
            param.requires_grad = True
        self.fix_classifier = False
        print(F'Free_Classifier: Done (training={self.classifier.training}, fix_classifier={self.fix_classifier})')
        #print(F'{self.classifier} --> freed')

    def ListCustomModules(self):
        for m in self.classifier.modules():
            if isinstance(m, SOW_V2) or isinstance(m, SOW_V1) or isinstance(m, SOW_V0) : yield m

    def IndicesSOW(self):
        for i, m in enumerate(self.classifier.modules()):
            if isinstance(m, SOW_V2) or isinstance(m, SOW_V1) or isinstance(m, SOW_V0) : yield i

    @classmethod
    def extra_parser(self, args):
        parser = ArgumentParser()
        parser.add_argument(
            "--pretrained-path",
            type=str,
            help="pretrained model path (use with --pretrained) (default: %(default)s)",
        )
        parser.add_argument(
            "--load-features",
            action='store_true',
            required=False,
            help="Load feature parameters from pretrained model, or no load. (default: %(default)s)",
        )
        parser.add_argument(
            "--fix-features",
            action='store_true',
            required=False,
            help="Fix not to calculate feature gradients when load pretrained model, or no fix. (default: %(default)s)",
        )
        parser.add_argument("--dropout", nargs="+", type=float)   # dropoutの複数指定対応
        parser.add_argument("--GS-rank", type=int, default=1024)
        return parser.parse_known_args(args)

def alexnet_32sow(device=None, **kwargs):
    return AlexNet_32SOW(device, **kwargs)
