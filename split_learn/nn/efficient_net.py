import torchvision.models
from torch import nn
from torchvision.models import efficientnet_b0


def make_efficient_net_b0_64x64_modules(num_classes: int = 200, pretrained: bool = False):
    """

    Architecture:
    Input           3 x 224 x 224
    ===================================
    First conv      32 x 112 x 112
    MBConv1         16 x 112 x 112
    MBConv6         24 x 56 x 56
    MBConv6         40 x 28 x 28
    MBConv6         80 x 14 x 14
    MBConv6         112 x 14 x 14
    MBConv6         192 x 7 x 7
    MBConv6         320 x 7 x 7
    LastConv        1280 x 2 x 2

    AvgPool         1280
    Linear          100

    :return:
    """
    if pretrained:
        b0_model = efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
    else:
        b0_model = efficientnet_b0()
    return list(b0_model.features.children()) + [nn.Sequential(b0_model.avgpool, nn.Flatten(1)), nn.Linear(1280, num_classes)]
