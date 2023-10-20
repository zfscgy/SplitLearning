from typing import Callable
import os
from functools import partial
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

DataLoader = partial(DataLoader,  num_workers=6, prefetch_factor=8, pin_memory=True, persistent_workers=True)

import torch.nn.functional as F

from deep_utils.datasets import Cifar100, TinyImageNet, YooChoose64, DBPedia
from deep_utils.data import random_split
from deep_utils.pretrained import get_glove_60B50d_word_embedding
from deep_utils.metrics import ClassAcc, HitRatio, Sparsity, MultiOutputMetric

from split_learn.modifiers import TruncationModifier
from split_learn.models import MultiOutputModelFactory, \
    MultiOutputResnet20Factory, MultiOutputEfficientNetFactory, MultiOutputGRU4RecFactory, MultiOutputTextCNNFactory, \
    make_resnet20_modules, make_gru4rec_modules, make_efficient_net_b0_64x64_modules
from split_learn.nn.gru4rec import bpr_loss
from split_learn.train import SplitTask

parser = argparse.ArgumentParser("Input index")
parser.add_argument("-i", nargs=1, type=int, dest="index")
args = parser.parse_args()
if args.index is not None:
    index = args.index[0]
else:
    index = 0
print("Index:", index)


TruncationModifier = partial(TruncationModifier, threshold=0)


class L1RegLoss:
    def __init__(self, original_loss: Callable, l1_coef: float):
        self.original_loss = original_loss
        self.l1_coef = l1_coef

    def __call__(self, model_output, label):
        y_pred, h = model_output
        return self.original_loss(y_pred, label) + self.l1_coef * F.l1_loss(h, torch.zeros_like(h), reduction='sum')


CrossEntropyWithL1 = partial(
    L1RegLoss,
    original_loss=nn.CrossEntropyLoss()
)


# cifar100_train, cifar100_test = Cifar100.get(txs_train=Cifar100.get_txs_random_transform())
# cifar100_train, cifar100_val = random_split(cifar100_train, 0.9)
# Cifar100L1Task = partial(
#     SplitTask,
#     model_factory=MultiOutputResnet20Factory(100),
#     get_modifier=TruncationModifier, split_layer=5,
#     train_loader=DataLoader(cifar100_train, 32),
#     validation_loader=DataLoader(cifar100_val, 256), test_loader=DataLoader(cifar100_test, 256),
#     get_optimizer=lambda ps: SGD(ps, 0.01, momentum=0.9),
#     metrics=[MultiOutputMetric(ClassAcc(), 0, 0), MultiOutputMetric(Sparsity(), 1, 0)], n_rounds=10000
# )

# yoochoose_train, yoochoose_test = YooChoose64.get(input_len=10)
# yoochoose_test, yoochoose_val = random_split(yoochoose_test, 0.5)
# # Since we don't want data argumentation in validation set
# YooChooseL1Task = partial(
#     SplitTask,
#     model_factory=MultiOutputGRU4RecFactory(YooChoose64.n_items, 32, 300),
#     get_modifier=TruncationModifier, split_layer=1,
#     train_loader=DataLoader(yoochoose_train, 64, shuffle=True),
#     validation_loader=DataLoader(yoochoose_val, 256), test_loader=DataLoader(yoochoose_test, 256),
#     get_optimizer=lambda ps: Adam(ps),
#     metrics=[MultiOutputMetric(HitRatio(20), 0, 0), MultiOutputMetric(Sparsity(), 1, 0)], n_rounds=10000,
# )



#
tinyImagenet_train, tinyImagenet_test = TinyImageNet.get(use_argumentation=True)
tinyImagenet_train, tinyImagenet_val = random_split(tinyImagenet_train, 0.9)
TinyImagenetL1Task = partial(
    SplitTask,
    model_factory=MultiOutputEfficientNetFactory(200, pretrained=False),
    get_modifier=TruncationModifier, split_layer=10,
    train_loader=DataLoader(tinyImagenet_train, 64),
    validation_loader=DataLoader(tinyImagenet_val, 256), test_loader=DataLoader(tinyImagenet_test, 256),
    get_optimizer=lambda ps: SGD(ps, 0.01, momentum=0.9),
    metrics=[MultiOutputMetric(ClassAcc(), 0, 0), MultiOutputMetric(Sparsity(), 1, 0)], n_rounds=10000,
)


# dbpedia_train, dbpedia_val, dbpedia_test = DBPedia.get(100)
#
# DBPediaL1Task = partial(
#     SplitTask,
#     model_factory=MultiOutputTextCNNFactory(DBPedia.n_vocabs, DBPedia.n_labels, 100, 50, 200, [3, 4, 5],
#                                             get_glove_60B50d_word_embedding(), DBPedia.word2idx),
#     get_modifier=TruncationModifier, split_layer=1,
#     train_loader=DataLoader(dbpedia_train, 64),
#     validation_loader=DataLoader(dbpedia_val, 256), test_loader=DataLoader(dbpedia_test, 256),
#     get_optimizer=lambda ps: Adam(ps),
#     metrics=[MultiOutputMetric(ClassAcc(), 0, 0), MultiOutputMetric(Sparsity(), 1, 0)], n_rounds=1000, batches_per_round=1000,
# )



l1_coefs = [0.0001, 0.0002]


if __name__ == '__main__':
    dir = "l1_tasks"
    if not Path(dir).is_dir():
        Path(dir).mkdir()
    os.chdir(dir)

    os.chdir("tinyImagenet")

    for seed in [1926, 1927, 1928]:
       TinyImagenetL1Task(loss_func=CrossEntropyWithL1(l1_coef=l1_coefs[index]), seed=seed,
                          prefix=f"tinyImagenet_l1_{l1_coefs[index]}").run()

    # TinyImagenetL1Task(loss_func=CrossEntropyWithL1(l1_coef=l1_coefs[index]), prefix=f"tinyImagenet_l1_{l1_coefs[index]}").run()

"""

export CUDA_VISIBLE_DEVICES=0
nohup python -u l1_tasks.py -i 0 > 0.log &
nohup python -u l1_tasks.py -i 1 > 1.log &
nohup python -u l1_tasks.py -i 2 > 2.log &
nohup python -u l1_tasks.py -i 3 > 3.log &
nohup python -u l1_tasks.py -i 4 > 4.log &


export CUDA_VISIBLE_DEVICES=1
nohup python -u l1_tasks.py -i 5 > 5.log &
nohup python -u l1_tasks.py -i 6 > 6.log &
nohup python -u l1_tasks.py -i 7 > 7.log &
nohup python -u l1_tasks.py -i 8 > 8.log &
nohup python -u l1_tasks.py -i 9 > 9.log &
"""