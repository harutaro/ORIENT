from argparse import Namespace

from torchvision import models

### Namespace of custom models provided by FACIL
from .lenet import LeNet
from .vggnet import vggnet, VggNet
from .resnet32 import resnet32, ResNet
### Namespace of custom models provided by KDDI
from .alexnet32 import alexnet_32, AlexNet_32
from .alexnet32sow import alexnet_32sow, AlexNet_32SOW
from .alexnet32supsup import alexnet_32supsup, AlexNet_32SUPSUP
from .alexnet32wsn import alexnet_32wsn, AlexNet_32WSN
from .resnet50 import resnet50_32, ResNet50_32
from .resnet50sow import resnet50_32sow, ResNet50_32SOW
from .resnet50supsup import resnet50_32supsup, ResNet50_32SUPSUP
from .resnet50wsn import resnet50_32wsn, ResNet50_32WSN

# available torchvision models
tvmodels = ['alexnet',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'googlenet',
            'inception_v3',
            'mobilenet_v2',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'wide_resnet50_2', 'wide_resnet101_2'
            ]

allmodels = tvmodels + [
    # Custom models difined by FACIL
    'resnet32',
    'LeNet',
    'vggnet',
    # Custom models defined by KDDI Research, Inc.
    'alexnet_32',
    'alexnet_32sow',
    'alexnet_32supsup',
    'alexnet_32wsn',
    'resnet50_32',
    'resnet50_32sow',
    'resnet50_32supsup',
    'resnet50_32wsn'
]


def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == models.Inception3:
        model.head_var = 'fc'
    elif type(model) == models.ResNet:
        model.head_var = 'fc'
    elif type(model) == models.VGG:
        model.head_var = 'classifier'
    elif type(model) == models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == models.SqueezeNet:
        model.head_var = 'classifier'
    else:
        raise ModuleNotFoundError


def extra_parser(network: str, args: list):
    '''実行時の引数を拡張する

    Args:
        network (str): ネットワーク名
        args (list[str]): コマンドライン引数

    Returns:
        tuple[Namespace, list[str]]: コマンドライン引数のNamespaceと値のリスト
    '''
    if network == 'alexnet_32':
        return AlexNet_32.extra_parser(args)
    elif network == 'alexnet_32supsup':
        return AlexNet_32SUPSUP.extra_parser(args)
    elif network == 'alexnet_32wsn':
        return AlexNet_32WSN.extra_parser(args)
    elif network == 'alexnet_32sow':
        return AlexNet_32SOW.extra_parser(args)
    elif network == 'resnet50_32':
        return ResNet50_32.extra_parser(args)
    elif network == 'resnet50_32supsup':
        return ResNet50_32SUPSUP.extra_parser(args)
    elif network == 'resnet50_32wsn':
        return ResNet50_32WSN.extra_parser(args)
    elif network == 'resnet50_32sow':
        return ResNet50_32SOW.extra_parser(args)
    else:
        return Namespace(), args
