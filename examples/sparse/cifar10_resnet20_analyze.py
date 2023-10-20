from typing import List
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from deep_utils.datasets import Cifar10
from deep_utils.convert import Convert
from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, SplitTrain, gen_param_str
from split_learn.nn.resnet import make_resnet20_modules
from split_learn.modifiers import TopKModifier, Modifier
from split_learn.analyze import NeuronInteraction, RoundMetricCallbacks
from split_learn.data import SplitXDataset

from deep_utils.ops import clone_module
from deep_utils.multi_process_run import parallel_process

import matplotlib.pyplot as plt

train_set, test_set = Cifar10.get()
train_loader = DataLoader(train_set, 64)
test_loader = DataLoader(test_set, 512)
split_train_set = SplitXDataset(train_set)
split_test_set = SplitXDataset(test_set)
split_train_loader = DataLoader(split_train_set, 64)
split_test_loader = DataLoader(split_test_set, 512)

resnet20_modules = make_resnet20_modules()


def plot_cut_layer(split_train: SplitTrain):
    if split_train.current_round % 10 != 0:
        return

    cut_layer_output = Convert.to_numpy(split_train.model.bottom_outputs[0][0].sum(dim=0))
    modified_output = Convert.to_numpy(split_train.model.detached_bottom_outputs[0][0].sum(dim=0))

    plt.imshow(cut_layer_output)
    plt.title(f"Original: {split_train.current_round}")
    plt.show()

    plt.imshow(modified_output)
    plt.title(f"Modified: {split_train.current_round}")
    plt.show()


def acc(y_pred, y_true):
    return np.mean(np.argmax(y_pred, -1) == y_true)


def centralized_train():
    modules = make_resnet20_modules()
    for module0, module in zip(resnet20_modules, modules):
        clone_module(module0, module)

    model = Convert.model_to_device(nn.Sequential(*modules))
    train = Train(model, train_loader, test_loader, {"acc": acc}, nn.CrossEntropyLoss(), lambda ps: Adam(ps))
    train.train(1000, 100, record_name=gen_param_str("cifar_resnet", {"optim": "adam"}))


def record_neuron_interaction(lr: float):
    modules = make_resnet20_modules()
    for module0, module in zip(resnet20_modules, modules):
        clone_module(module0, module)
    bottom_model = Convert.model_to_device(nn.Sequential(*modules[:2]))
    top_model = Convert.model_to_device(nn.Sequential(nn.BatchNorm2d(32), *modules[2:]))
    modifier = Modifier()
    model = TwoLayerSplitModel([bottom_model], top_model, lambda xs: xs[0], modifiers=[modifier])

    neuron_interaction = NeuronInteraction(bottom_model, train_set)
    record_name = gen_param_str("analyze-cifar_resnet", {"lr": lr, "optim": "sgd"})
    metric_callbacks = RoundMetricCallbacks(record_name + "-round_metrics", {
        "inSelf": lambda _: neuron_interaction.get_in_sample_self(),
        "inCross": lambda _: neuron_interaction.get_in_sample_cross(),
        "crossSelf": lambda _: neuron_interaction.get_cross_sample_self(),
        "crossCross": lambda _: neuron_interaction.get_cross_sample_cross()
    })

    train = SplitTrain(model, split_train_loader, split_test_loader, {"acc": acc},
                       nn.CrossEntropyLoss(), lambda ps: SGD(ps, lr, momentum=0.9))
    train.train(1000, 100, round_callback=metric_callbacks, record_name=record_name)


if __name__ == '__main__':
    seed = 20220307001
    torch.manual_seed(seed)
    os.chdir("Cifar10-20220307001")
    record_neuron_interaction(0.01)
