import os
from typing import List
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from deep_utils.datasets import Cifar10
from deep_utils.convert import Convert

from split_learn.data import SplitXDataset
from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, SplitTrain, gen_param_str
from split_learn.nn.resnet import make_resnet20_modules
from split_learn.modifiers import TopKModifier, RandomDropoutModifier, FixedSparseModifier, RandomTopKModifier

from deep_utils.ops import clone_module
from deep_utils.multi_process_run import parallel_process

train_set_raw, test_set_raw = Cifar10.get()
train_loader_raw = DataLoader(train_set_raw, 64)
test_loader_raw = DataLoader(test_set_raw, 512)

train_set, test_set = SplitXDataset(train_set_raw), SplitXDataset(test_set_raw)
train_loader = DataLoader(train_set, 64)
test_loader = DataLoader(test_set, 64)

seed = 20220307001
torch.manual_seed(seed)
resnet20_modules = make_resnet20_modules()


def acc(y_pred, y_true):
    return np.mean(np.argmax(y_pred, -1) == y_true)


def centralized_train(lr: float, n_rounds: int = 1000, n_batches_per_round: int = 100):
    modules = make_resnet20_modules()
    for module0, module in zip(resnet20_modules, modules):
        clone_module(module0, module)

    model = Convert.model_to_device(nn.Sequential(*modules))
    train = Train(model, train_loader_raw, test_loader_raw, {"acc": acc}, nn.CrossEntropyLoss(), lambda ps: SGD(ps, lr, momentum=0.9))
    train.train(n_rounds, n_batches_per_round, record_name=gen_param_str("cifar_resnet", {"optim": "sgd", "lr": lr}),
                save_model_n_round=(100, 100))


def split_train_sparse(sparsity: float, lr: float, modifier: str, n_rounds: int = 1000, n_batches_per_round: int = 100):
    modules = make_resnet20_modules()
    for module0, module in zip(resnet20_modules, modules):
        clone_module(module0, module)

    bottom_models = [Convert.model_to_device(nn.Sequential(*modules[:2]))]
    top_model = Convert.model_to_device(nn.Sequential(*modules[2:]))

    get_modifier = {
        "topk": TopKModifier,
        "randTopk": RandomTopKModifier,
        "randTopk2": lambda rate: RandomTopKModifier(rate, 2),
        "randTopk4": lambda rate: RandomTopKModifier(rate, 4),
        "random": RandomDropoutModifier,
        "fixed": lambda rate: FixedSparseModifier(rate, [32, 32, 32])
    }[modifier]
    model = TwoLayerSplitModel(bottom_models, top_model, lambda x: x[0], [get_modifier(sparsity)])

    train = SplitTrain(model, train_loader, test_loader, {"acc": acc},
                       nn.CrossEntropyLoss(), lambda ps: SGD(ps, lr, momentum=0.9))

    record_name = gen_param_str("cifar_resnet", {"modifier": modifier, "sparsity": sparsity, "optim": "sgd", "lr": lr})
    train.train(n_rounds, n_batches_per_round, record_name=record_name, save_model_n_round=(100, 100))


if __name__ == '__main__':
    save_dir = Path(f"Cifar10-{seed}")
    if not save_dir.is_dir():
        save_dir.mkdir()
    os.chdir(save_dir)
    # split_train_sparse(0.1, 0.01, "random", 2000, 200)
    parallel_process([
        # lambda: centralized_train(0.01, 1000, 100),
        # lambda: split_train_sparse(0.1, 0.01, "randTopk", 3000, 200),
        # lambda: split_train_sparse(0.05, 0.01, "randTopk", 4000, 200),
        # lambda: split_train_sparse(0.02, 0.01, "randTopk", 4000, 200),
        # lambda: split_train_sparse(0.1, 0.01, "topk", 3000, 200),
        # lambda: split_train_sparse(0.05, 0.01, "topk", 4000, 200),
        # lambda: split_train_sparse(0.02, 0.01, "topk", 4000, 200),
        # lambda: split_train_sparse(0.1, 0.01, "fixed", 2000, 200),
        # lambda: split_train_sparse(0.05, 0.01, "fixed", 2000, 200),
        # lambda: split_train_sparse(0.02, 0.01, "fixed", 2000, 200),
        # lambda: split_train_sparse(0.1, 0.01, "random", 2000, 200),
        # lambda: split_train_sparse(0.05, 0.01, "random", 2000, 200),
        # lambda: split_train_sparse(0.02, 0.01, "random", 2000, 200),

        lambda: split_train_sparse(0.1, 0.01, "randTopk2", 3000, 200),
        lambda: split_train_sparse(0.05, 0.01, "randTopk2", 4000, 200),
        lambda: split_train_sparse(0.02, 0.01, "randTopk2", 4000, 200),

        lambda: split_train_sparse(0.1, 0.01, "randTopk", 3000, 200),
        lambda: split_train_sparse(0.05, 0.01, "randTopk", 4000, 200),
        lambda: split_train_sparse(0.02, 0.01, "randTopk", 4000, 200),

        lambda: split_train_sparse(0.1, 0.01, "randTopk4", 3000, 200),
        lambda: split_train_sparse(0.05, 0.01, "randTopk4", 4000, 200),
        lambda: split_train_sparse(0.02, 0.01, "randTopk4", 4000, 200),
    ], ["cuda:0"] * 5 + ["cuda:1"] * 4)

