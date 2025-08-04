from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

## for SOW class
import math
from typing import Any, Tuple
from torch import Tensor
from torch.autograd import Function
import numpy as np
import scipy.optimize as op

from .wsn_libs import Wsn_SubnetLinear

__all__ = ("alexnet_32wsn")

class AlexNet_32WSN(nn.Module):
    ''' input size: 3 channels x 32 pixels x 32 pixels '''
    ''' replaced nn.Linears with Wsn_SubnetLinears in the classifier block. '''

    def __init__(self, device='cuda', num_classes=1000, dropout=0.5,
                 fix_features=True, load_features=False, pretrained_path='',
                 num_tasks=10, **kwargs):
        super().__init__()

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
            Wsn_SubnetLinear(256*4*4, 4096, bias=False, num_tasks=num_tasks, device=device, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(p=d2),
            Wsn_SubnetLinear(4096, 4096, bias=False, num_tasks=num_tasks, device=device, dtype=torch.float64),
            nn.ReLU(),
        )
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(4096, num_classes, bias=True)
        #self.fc.weight.requires_grad = False
        #self.fc.bias.requires_grad = False
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
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.fix_classifier = True
        print(F'Fix_Classifier: Done (training={self.classifier.training}, fix_classifier={self.fix_classifier})')
        #print(F'{self.classifier} --> freezed')

    def Free_Classifier(self):
        self.classifier.train()
        for name, param in self.classifier.named_parameters():
            if name.endswith('.fix_s') or name.endswith('.fix_matrix') :
                continue # for saving fix_s
            param.requires_grad = True
        self.fix_classifier = False
        print(F'Free_Classifier: Done (training={self.classifier.training}, fix_classifier={self.fix_classifier})')
        #print(F'{self.classifier} --> freed')

    def ListCustomModules(self):
        for m in self.classifier.modules():
            if isinstance(m, Wsn_SubnetLinear) : yield m

    def IndicesWsn(self):
        for i, m in enumerate(self.classifier.modules()):
            if isinstance(m, Wsn_SubnetLinear) : yield i

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
        return parser.parse_known_args(args)

    def save_pretrained(self, save_path):
        state_dict = self.features.state_dict()
        torch.save(state_dict, save_path)

def alexnet_32wsn(device=None, **kwargs):
    return AlexNet_32WSN(device, **kwargs)
