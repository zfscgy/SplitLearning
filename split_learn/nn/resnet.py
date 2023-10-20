import time
from typing import List, Tuple, Type
import numpy as np
import torch.random
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        """
        :param x:  [..., in_channels, H, W]
        :return:   [..., out_channels, H', W']  W' = W / stride, H' = H / stride
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        """
        :param x: [..., in_channels, H, W]
        :return: [..., out_channels * 4, H', W']  W' = W / stride, H' = H / stride
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def make_layer(block_class, in_channels, out_channels, n_blocks: int, stride: int):
    """
    :param block_class:
    :param out_channels:
    :param n_blocks:
    :param stride:
    :return: Module with input [... in_channels, H, W],
        and output [... out_channels, H', W']  W' = W / stride, H' = H / stride
    """
    channels = in_channels
    strides = [stride] + [1] * (n_blocks - 1)  # Only the first block use the stride parameter
    layers = []
    for stride in strides:
        layers.append(block_class(channels, out_channels, stride))
        channels = out_channels * block_class.expansion

    return nn.Sequential(*layers)


def make_resenet_modules(initial_shape: List[int], layers: List[Tuple[int, int, Type]], output_dim: int):
    c0, h0, w0 = initial_shape

    first_conv_layer = nn.Sequential(nn.Conv2d(c0, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
    # [16, h0, w0]

    current_channels = 16
    current_h, current_w = h0, w0

    res_layers = []
    for layer_param in layers:
        n_blocks, stride, block_class = layer_param
        res_layers.append(make_layer(block_class, current_channels, current_channels * 2, n_blocks, stride))
        current_channels *= 2
        current_h //= stride
        current_w //= stride

    avg_pool_layer = nn.Sequential(nn.AvgPool2d((current_h, current_w)), nn.Flatten())
    last_dense_layer = nn.Linear(current_channels, output_dim)
    return [first_conv_layer] + res_layers + [avg_pool_layer, last_dense_layer]


def make_resnet20_modules(outdim: int = 10):
    """
    Architecture:
    Input                3 x 32 x 32
    =================================
    First-conv          16 x 32 x 32
    BasicBlock1         32 x 32 x 32
    BasicBlock2         64 x 16 x 16
    BasicBlock3         128 x 8 x 8
    AvgPool             128
    Linear              10
    :return:
    """
    return make_resenet_modules(
        [3, 32, 32],
        [(3, 1, BasicBlock), (3, 2, BasicBlock), (3, 2, BasicBlock)],  # out_channels = 16 * 2 * 2 = 64
        outdim
    )
