import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from deep_utils.datasets import Mnist
from deep_utils.convert import Convert


from split_learn.data import SplitXDataset

from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, SplitTrain, gen_param_str
from split_learn.modifiers import FixedSparseModifier, RandomDropoutModifier, TopKModifier, \
    RandomTopKModifier

from deep_utils.ops import clone_module
from deep_utils.multi_process_run import parallel_process

train_set_raw, test_set_raw = Mnist.get(txs=[Mnist.tx_flatten])
train_set = SplitXDataset(train_set_raw)
test_set = SplitXDataset(test_set_raw)

train_loader_raw = DataLoader(train_set_raw, 32)
test_loader_raw = DataLoader(test_set_raw, 512)

train_loader = DataLoader(train_set, 32)
test_loader = DataLoader(test_set, 512)


seed = 20220307001
torch.manual_seed(seed)
save_dir = Path(f"Mnist-{seed}")
if not save_dir.is_dir():
    save_dir.mkdir()
os.chdir(save_dir)


dnn_modules = [
    nn.Sequential(nn.Linear(784, 128), nn.ReLU()),
    nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
    nn.Linear(32, 10)
]


def acc(y_pred, y_true):
    return np.mean(np.argmax(y_pred, -1) == y_true)


def centralized_train(lr: float, n_rounds: int = 500, n_batches_per_round: int = 100):
    modules = [
        nn.Sequential(nn.Linear(784, 128), nn.Sigmoid()),
        nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
        nn.Linear(32, 10)
    ]
    for module0, module in zip(dnn_modules, modules):
        clone_module(module0, module)

    model = Convert.model_to_device(nn.Sequential(*modules))

    train = Train(model, train_loader_raw, test_loader_raw, {"acc": acc}, nn.CrossEntropyLoss(), lambda ps: SGD(ps, lr, momentum=0.9))
    train.train(n_rounds, n_batches_per_round, record_name=gen_param_str("mnist_dnn", {"optim": "sgd", "lr": lr}),
                save_model_n_round=(100, 100))


def split_train_sparse(sparsity: float, lr: float, modifier: str = 'topk', n_rounds: int = 500, n_batches_per_round: int = 100):
    modules = [
        nn.Sequential(nn.Linear(784, 128), nn.Sigmoid()),
        nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
        nn.Linear(32, 10)
    ]

    for module0, module in zip(dnn_modules, modules):
        clone_module(module0, module)

    get_modifier = {
        "topk": TopKModifier,
        "randTopk": RandomTopKModifier,
        "randTopk2": lambda rate: RandomTopKModifier(rate, 2),
        "randTopk4": lambda rate: RandomTopKModifier(rate, 4),
        "randTopk5": lambda rate: RandomTopKModifier(rate, 5),
        "random": RandomDropoutModifier,
        "fixed": lambda rate: FixedSparseModifier(rate, [128])
    }[modifier]

    model = TwoLayerSplitModel(
        [Convert.model_to_device(modules[0])],
         Convert.model_to_device(nn.Sequential(*modules[1:])),
        modifiers=[get_modifier(sparsity)]
    )
    train = SplitTrain(model, train_loader, test_loader, {"acc": acc}, nn.CrossEntropyLoss(),
                       lambda m: SGD(m, lr, momentum=0.9))

    record_name = gen_param_str(
        "mnist_dnn", {"modifier": modifier, "sparsity": sparsity, "optim": "sgd", "lr": lr})

    train.train(n_rounds, n_batches_per_round, record_name=record_name, save_model_n_round=(100, 100))


if __name__ == '__main__':
    parallel_process([
        lambda: centralized_train(0.01, 1000, 100),
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
        # lambda: split_train_sparse(0.1, 0.01, "randTopk2", 3000, 200),
        # lambda: split_train_sparse(0.05, 0.01, "randTopk2", 4000, 200),
        # lambda: split_train_sparse(0.02, 0.01, "randTopk2", 4000, 200),
        # lambda: split_train_sparse(0.1, 0.01, "randTopk4", 3000, 200),
        # lambda: split_train_sparse(0.05, 0.01, "randTopk4", 4000, 200),
        # lambda: split_train_sparse(0.02, 0.01, "randTopk4", 4000, 200),
        # lambda: split_train_sparse(0.1, 0.01, "randTopk5", 3000, 200),
        # lambda: split_train_sparse(0.05, 0.01, "randTopk5", 4000, 200),
        # lambda: split_train_sparse(0.02, 0.01, "randTopk5", 4000, 200),
    ], ["cuda:1"])


