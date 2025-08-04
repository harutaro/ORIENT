from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

## for SOW class
import math
from typing import Any, Tuple, Union
from torch import Tensor
from torch.autograd import Function
import numpy as np
import scipy.optimize as op

from .wsn_libs import Wsn_SubnetLinear

__all__ = ('resnet50_32wsn')


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x: Tensor) -> Tensor:
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers"""

    expansion = 2

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BottleNeck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BottleNeck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x: Tensor) -> Tensor:
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet50_32WSN(nn.Module):
    ''' input size: 3 channels x 32 pixels x 32 pixels '''
    ''' replaced nn.Linear with Wsn_SubnetLinear in the classifier block.'''

    def __init__(self, block: Union[BasicBlock, BottleNeck], num_block: list,
                 device='cuda', num_classes=1000, dropout=0.5,
                 fix_features=True, load_features=False, pretrained_path='',
                 num_tasks=10, **kwargs):
        super().__init__()

        self.fix_features = fix_features
        self.fix_classifier = False
        self.in_channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            self._make_layer(block, 64, num_block[0], 1),  # 64,3
            self._make_layer(block, 128, num_block[1], 2),  # 128,4
            self._make_layer(block, 256, num_block[2], 2),  # 256,6
            self._make_layer(block, 512, num_block[3], 2),  # 512,3
            # we use a different inputsize than the original paper
            # so conv2_x's stride is 1
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            Wsn_SubnetLinear(512*block.expansion, 512*block.expansion, bias=False, num_tasks=num_tasks, device=device, dtype=torch.float64),
            nn.ReLU(),  # inplace=False
        )  # 512,2
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)
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

    def _make_layer(
        self,
        block: Union[BasicBlock, BottleNeck],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float64)  # to float64
        x = self.classifier(x)
        x = x.to(dtype=torch.float)    # to float
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
        def sub_Fix_Features(m):
            if isinstance(m, nn.BatchNorm2d): m.track_running_stats = False
            for m2 in m.children(): sub_Fix_Features(m2)
        sub_Fix_Features(self.features)
        self.fix_features = True
        print("Fix_Features: Done")

    def Fix_Classifier(self):
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.fix_classifier = True
        print(F'Fix_Classifier: Done (training={self.classifier.training}, fix_classifier={self.fix_classifier})')

    def Free_Classifier(self):
        self.classifier.train()
        for name, param in self.classifier.named_parameters():
            if name.endswith('.fix_s') or name.endswith('.fix_matrix'):
                continue
            param.requires_grad = True
        self.fix_classifier = False
        print(F'Free_Classifier: Done (training={self.classifier.training}, fix_classifier={self.fix_classifier})')

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
        parser.add_argument("--dropout", type=float, default=0.5)
        return parser.parse_known_args(args)

    def save_pretrained(self, save_path):
        state_dict = self.features.state_dict()
        torch.save(state_dict, save_path)

def resnet50_32wsn(device=None, **kwargs):
    return ResNet50_32WSN(BottleNeck, [3, 4, 6, 3], device, **kwargs)
