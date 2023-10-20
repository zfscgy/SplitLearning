import os
from functools import partial
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

DataLoader = partial(DataLoader,  num_workers=2, prefetch_factor=8, pin_memory=True, persistent_workers=True)


import torch.nn.functional as F


from deep_utils.datasets import Mnist, Cifar10, YooChoose64, CriteoSmall, Cifar100, TinyImageNet, DBPedia
from deep_utils.data import random_split
from deep_utils.metrics import ClassAcc, Auc, HitRatio
from deep_utils.pretrained import get_glove_60B50d_word_embedding

from split_learn.modifiers import Modifier, FixedSparseModifier, TopKModifier, RandomDropoutModifier, \
    NaiveRandomTopKModifier, RandomTopKModifier, UniformRandomTopKModifier, \
    QuantizationModifier
from split_learn.models import MnistDNNFactory, Resnet20Factory, EfficientNetFactory, GRU4RecFactory, TextCNNFactory
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


# cifar100_train, cifar100_test = Cifar100.get(txs_train=Cifar100.get_txs_random_transform())
# cifar100_train, cifar100_val = random_split(cifar100_train, 0.9)

tinyImagenet_train, tinyImagenet_test = TinyImageNet.get(use_argumentation=True)
tinyImagenet_train, tinyImagenet_val = random_split(tinyImagenet_train, 0.9)

# yoochoose_train, yoochoose_test = YooChoose64.get(input_len=10)
# yoochoose_test, yoochoose_val = random_split(yoochoose_test, 0.5)
# Since we don't want data argumentation in validation set

#
# Cifar100Task = partial(
#     SplitTask,
#     model_factory=Resnet20Factory(outdim=100),
#     train_loader=DataLoader(cifar100_train, 32),
#     validation_loader=DataLoader(cifar100_val, 256), test_loader=DataLoader(cifar100_test, 256),
#     loss_func=nn.CrossEntropyLoss(), get_optimizer=lambda ps: SGD(ps, 0.01, momentum=0.9),
#     metrics=[ClassAcc()], n_rounds=10000,
#     prefix="cifar100"
# )

TinyImagenetTask = partial(
    SplitTask,
    model_factory=EfficientNetFactory(outdim=200),
    train_loader=DataLoader(tinyImagenet_train, 64),
    validation_loader=DataLoader(tinyImagenet_val, 256), test_loader=DataLoader(tinyImagenet_test, 256),
    loss_func=nn.CrossEntropyLoss(), get_optimizer=lambda ps: SGD(ps, 0.01, momentum=0.9),
    metrics=[ClassAcc()], n_rounds=10000,
    prefix="tinyImagenet"
)
#
#
# YooChooseTask = partial(
#     SplitTask,
#     model_factory=GRU4RecFactory(YooChoose64.n_items, 32, 300, additional_linear=None),
#     train_loader=DataLoader(yoochoose_train, 64, shuffle=True),
#     validation_loader=DataLoader(yoochoose_val, 256), test_loader=DataLoader(yoochoose_test, 256),
#     loss_func=nn.CrossEntropyLoss(), get_optimizer=lambda ps: Adam(ps),
#     metrics=[HitRatio(20)], n_rounds=10000,
#     prefix="yoochoose"
# )


# dbpedia_train, dbpedia_val, dbpedia_test = DBPedia.get(100)
#
# DBPediaTask = partial(
#     SplitTask,
#     model_factory=TextCNNFactory(DBPedia.n_vocabs, DBPedia.n_labels, 100, 50, 200, [3, 4, 5],
#                                  get_glove_60B50d_word_embedding(), DBPedia.word2idx),
#     train_loader=DataLoader(dbpedia_train, 64),
#     validation_loader=DataLoader(dbpedia_val, 256), test_loader=DataLoader(dbpedia_test, 256),
#     loss_func=nn.CrossEntropyLoss(), get_optimizer=lambda ps: Adam(ps),
#     metrics=[ClassAcc()], n_rounds=1000, batches_per_round=1000,
#     prefix="dbpedia"
# )



if __name__ == '__main__':
    # seed = [1926, 1927, 1928, 1929, 1930][index]



    dir = "tinyImagenet"

    if not Path(dir).is_dir():
        Path(dir).mkdir()
    os.chdir(dir)



    modifier_settings_cifar = [
        (Modifier,),
        (FixedSparseModifier, 0.03),
        (FixedSparseModifier, 0.06),
        (FixedSparseModifier, 0.125),
        (TopKModifier, 0.02),
        (TopKModifier, 0.05),
        (TopKModifier, 0.1),
        # (QuantizationModifier, 1, False),
        (QuantizationModifier, 2, False),
        (QuantizationModifier, 3, False),
        (QuantizationModifier, 4, False),
        (NaiveRandomTopKModifier, 0.02, 0.05),
        (NaiveRandomTopKModifier, 0.05, 0.05),
        (NaiveRandomTopKModifier, 0.1,  0.05),
        (NaiveRandomTopKModifier, 0.02, 0.1),
        (NaiveRandomTopKModifier, 0.05, 0.1),
        (NaiveRandomTopKModifier, 0.1, 0.1),
        (NaiveRandomTopKModifier, 0.02, 0.2),
        (NaiveRandomTopKModifier, 0.05, 0.2),
        (NaiveRandomTopKModifier, 0.1, 0.2),
        (NaiveRandomTopKModifier, 0.02, 0.3),
        (NaiveRandomTopKModifier, 0.05, 0.3),
        (NaiveRandomTopKModifier, 0.1, 0.3),
    ]

    modifier_settings_tinyImagenet = [
        (Modifier,),
        (FixedSparseModifier, 3 / 1280),
        (FixedSparseModifier, 6 / 1280),
        (FixedSparseModifier, 12 / 1280),
        (TopKModifier, 2 / 1280),
        (TopKModifier, 4 / 1280),
        (TopKModifier, 9 / 1280),
        (NaiveRandomTopKModifier, 2 / 1280, 0.1),
        (NaiveRandomTopKModifier, 4 / 1280, 0.1),
        (NaiveRandomTopKModifier, 9 / 1280, 0.1),
    ]

    modifier_settings_yoochoose = [
        # (Modifier,),
        # (FixedSparseModifier, 3 / 300),
        # (FixedSparseModifier, 6 / 300),
        # (FixedSparseModifier, 12 / 300),
        # (TopKModifier, 2 / 300),
        # (TopKModifier, 4 / 300),
        # (TopKModifier, 9 / 300),
        (NaiveRandomTopKModifier, 2 / 300, 0.05),
        (NaiveRandomTopKModifier, 4 / 300, 0.05),
        (NaiveRandomTopKModifier, 9 / 300, 0.05),
        # (NaiveRandomTopKModifier, 2 / 300, 0.1),
        # (NaiveRandomTopKModifier, 4 / 300, 0.1),
        # (NaiveRandomTopKModifier, 9 / 300, 0.1),
        (NaiveRandomTopKModifier, 2 / 300, 0.2),
        (NaiveRandomTopKModifier, 4 / 300, 0.2),
        (NaiveRandomTopKModifier, 9 / 300, 0.2),
        (NaiveRandomTopKModifier, 2 / 300, 0.3),
        (NaiveRandomTopKModifier, 4 / 300, 0.3),
        (NaiveRandomTopKModifier, 9 / 300, 0.3),
        # (QuantizationModifier, 1),
        # (QuantizationModifier, 2)
    ]

    modifier_setting_dbpedia = [
        (Modifier,),
        (FixedSparseModifier, 3 / 600),
        (FixedSparseModifier, 6 / 600),
        (FixedSparseModifier, 12 / 600),
        (FixedSparseModifier, 18 / 600),
        (TopKModifier, 2 / 600),
        (TopKModifier, 4 / 600),
        (TopKModifier, 9 / 600),
        (TopKModifier, 14 / 600),
        (NaiveRandomTopKModifier, 2 / 600, 0.1),
        (NaiveRandomTopKModifier, 4 / 600, 0.1),
        (NaiveRandomTopKModifier, 9 / 600, 0.1),
        (NaiveRandomTopKModifier, 14 / 600, 0.1),
        (QuantizationModifier, 1),
        (QuantizationModifier, 2)
    ]

    #
    # modifier_settings = modifier_settings_cifar
    # Task = Cifar100Task
    # split_position = 5

    # modifier_settings = modifier_settings_yoochoose
    # Task = YooChooseTask
    # split_position = 1

    modifier_settings = modifier_settings_tinyImagenet
    Task = TinyImagenetTask
    split_position = 10

    # modifier_settings = modifier_setting_dbpedia
    # Task = DBPediaTask
    # split_position = 1


    task_ids = range(index, len(modifier_settings), 4)

    def run_sparse_train():
        for seed in [1926, 1927, 1928]:
            for i in task_ids:
                m = modifier_settings[i]
                if len(m) >= 2:
                    Task(
                        get_modifier=lambda input_shape: m[0](m[1], input_shape, *m[2:]),
                        split_layer=split_position, seed=seed).run()
                else:
                    Task(
                        get_modifier=lambda input_shape: m[0](input_shape),
                        split_layer=split_position, seed=seed).run()

    run_sparse_train()



