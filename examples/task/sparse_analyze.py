import os
from pathlib import Path
import argparse
from functools import partial

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from deep_utils.train import RoundMetricsRecorder
from deep_utils.datasets import Mnist, Cifar100
from deep_utils.data import random_split
from deep_utils.metrics import ClassAcc


from split_learn.train import SplitTask
from split_learn.analyze import TopkReplacements, NeuronInteraction, NeuronChange, OrderChange,  NeuronCoAdaption, \
    LastWeightNorm, LossOnTrainSet, MetricOnTrainSet, \
    SingleSampleLossChange
from split_learn.models import SequentialSplitFactory, Resnet20Factory
from split_learn.modifiers import Modifier, TopKModifier, NaiveRandomTopKModifier


parser = argparse.ArgumentParser("Input index")
parser.add_argument("-i", nargs=1, type=int, dest="index")
args = parser.parse_args()
if args.index is not None:
    index = args.index[0]
else:
    index = 0
print("Index:", index)



# mnist_train, mnist_test = Mnist.get(txs=[Mnist.tx_flatten])
# sparsity = 0.02
# split_layer = 2
# seed = 1926
#
#
# MnistAnalyzeTask = partial(
#     SplitTask,
#     model_factory=SequentialSplitFactory(lambda: [
#         nn.Sequential(nn.Linear(784, 128), nn.LeakyReLU()),
#         nn.Sequential(nn.Linear(128, 100), nn.Tanh()),
#         nn.Sequential(nn.Linear(100, 10))
#     ], [[128], [100]]),
#     train_loader=DataLoader(mnist_train, 32),
#     validation_loader=DataLoader(mnist_test, 512), test_loader=DataLoader(mnist_test, 512),
#     loss_func=nn.CrossEntropyLoss(), get_optimizer=lambda ps: SGD(ps, 0.01, momentum=0.9),
#     metrics=[ClassAcc()], n_rounds=10000,
#     prefix="mnist",
#     round_callback=RoundMetricsRecorder(
#         [
#             LossOnTrainset(),
#             OrderChange(sparsity),
#             TopkReplacements(sparsity),
#             NeuronChange(sparsity),
#             NeuronInteraction()
#         ]
#     )
# )



cifar100_train, cifar100_test = Cifar100.get(txs_train=Cifar100.get_txs_random_transform())

sparsity = 0.02
split_layer = 5
seed = 1926


Cifar100AnalyzeTask = partial(
    SplitTask,
    model_factory=Resnet20Factory(outdim=100),
    train_loader=DataLoader(cifar100_train, 32),
    validation_loader=DataLoader(cifar100_test, 256), test_loader=DataLoader(cifar100_test, 256),
    loss_func=nn.CrossEntropyLoss(), get_optimizer=lambda ps: SGD(ps, 0.01, momentum=0.9),
    metrics=[ClassAcc()], n_rounds=10000,
    prefix="cifar100",
    round_callback=RoundMetricsRecorder(
        [
            MetricOnTrainSet(),
            LossOnTrainSet(),
            # TopkReplacements(sparsity),
            # SingleSampleLossChange(TopKModifier(sparsity, [128])),
        ]
    ),
    save_model=True
)



modifier_settings = [
    (Modifier,),
    (TopKModifier, sparsity),
    (NaiveRandomTopKModifier, sparsity, 0.05),
    (NaiveRandomTopKModifier, sparsity, 0.1),
    (NaiveRandomTopKModifier, sparsity, 0.2),
]

Task = Cifar100AnalyzeTask


def run_sparse_analyzes():
    m = modifier_settings[index]
    if len(m) >= 2:
        Task(
            get_modifier=lambda input_shape: m[0](m[1], input_shape, *m[2:]),
            split_layer=split_layer, seed=seed).run()
    else:
        Task(
            get_modifier=lambda input_shape: m[0](input_shape),
            split_layer=split_layer, seed=seed).run()


if __name__ == '__main__':
    directory = "analyze-model"
    # directory = "test"
    if not Path(directory).is_dir():
        Path(directory).mkdir()
    os.chdir(directory)

    run_sparse_analyzes()
