from pathlib import Path
import os

from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset

from deep_utils.datasets import CriteoSmall
from deep_utils.convert import Convert
from deep_utils.ops import clone_module
from deep_utils.multi_process_run import parallel_process

from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, gen_param_str, SplitTrain
from split_learn.nn.wide_and_deep import WideAndDeepNet, DeepNet
from split_learn.modifiers import Modifier, FixedSparseModifier, RandomDropoutModifier, TopKModifier, RandomTopKModifier


class CriteoSmallDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        x_n, x_c, y = self.original_dataset[index]
        return [x_n, x_c], y

    def __len__(self):
        return len(self.original_dataset)


class CriteoSmallWide(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        x_n, x_c, y = self.original_dataset[index]
        return x_n, y

    def __len__(self):
        return len(self.original_dataset)


embedding_dim = 32
dnn_out_dim = 32
batch_size = 128
seed = 20220307001
torch.manual_seed(seed)


def get_dnn_modules():
    return [
        nn.Sequential(nn.Linear(len(CriteoSmall.category_counts) * embedding_dim, 256), nn.ReLU()),
        nn.Sequential(nn.Linear(256, 32), nn.ReLU())
    ]


def acc(y_pred, y_true):
    return np.mean(np.round(y_pred) == y_true)


def auc(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)


def only_wide_train(lr: float):
    train_set_raw, test_set_raw = CriteoSmall.get()
    train_set_wide = CriteoSmallWide(train_set_raw)
    test_set_wide = CriteoSmallWide(test_set_raw)
    train_loader_wide = DataLoader(train_set_wide, batch_size)
    test_loader_wide = DataLoader(test_set_wide, batch_size)


    model = Convert.model_to_device(nn.Sequential(nn.Linear(CriteoSmall.n_numerical_features, 1), nn.Sigmoid()))
    train = Train(model, train_loader_wide, test_loader_wide, {"acc": acc, "auc": auc}, F.binary_cross_entropy,
                  lambda ps: SGD(ps, lr, momentum=0.9))
    train.train(500, 100, record_name=gen_param_str("criteo_wide", {"optim": "sgd", "lr": lr}),
                save_model_n_round=(100, 100))


def centralized_train(lr: float, n_rounds: int = 500, n_batches_per_round: int = 100):
    train_set_raw, test_set_raw = CriteoSmall.get()
    train_set = CriteoSmallDataset(train_set_raw)
    test_set = CriteoSmallDataset(test_set_raw)
    model = Convert.model_to_device(
        WideAndDeepNet(CriteoSmall.n_numerical_features, CriteoSmall.category_counts,
        nn.Sequential(*get_dnn_modules()), dnn_out_dim, embedding_dim))
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, 128)

    print("Clone finished")
    train = Train(model, train_loader, test_loader, {"acc": acc, "auc": auc}, F.binary_cross_entropy,
                  lambda ps: SGD(ps, lr, momentum=0.9))

    train.train(n_rounds, n_batches_per_round, record_name=gen_param_str("criteo_wide-n-deep", {"optim": "sgd", "lr": lr}),
                save_model_n_round=(100, 100))


def split_train_sparse(sparsity: float, lr: float, modifier: str = 'topk', n_rounds: int = 500, n_batches_per_round: int = 100):
    train_set_raw, test_set_raw = CriteoSmall.get()
    train_set = CriteoSmallDataset(train_set_raw)
    test_set = CriteoSmallDataset(test_set_raw)
    model = Convert.model_to_device(
        WideAndDeepNet(
            CriteoSmall.n_numerical_features, CriteoSmall.category_counts,
            nn.Sequential(*get_dnn_modules()), dnn_out_dim, embedding_dim))

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, 128)


    split_position = 1
    deepnet_bottom = DeepNet(CriteoSmall.category_counts, nn.Sequential(*list(model.mlp.mlp.children())[:split_position]))
    deepnet_bottom.embedding_layers = model.mlp.embedding_layers

    class TopModel(nn.Module):
        def __init__(self):
            super(TopModel, self).__init__()
            self.dnn = nn.Sequential(*list(model.deep_model.mlp.children())[split_position:])
            self.prediction_model = model.prediction_model

        def forward(self, x):
            x_wide, x_deep = x
            deep_out = self.dnn(x_deep)
            last_feature = torch.cat([deep_out, x_wide], dim=-1)
            return torch.sigmoid(self.prediction_model(last_feature))

    get_modifier = {
        "topk": TopKModifier,
        "randTopk": RandomTopKModifier,
        "randTopk2": lambda rate: RandomTopKModifier(rate, 2),
        "randTopk4": lambda rate: RandomTopKModifier(rate, 4),
        "random": RandomDropoutModifier,
        "fixed": lambda rate: FixedSparseModifier(rate, [256])
    }[modifier]
    model = TwoLayerSplitModel([Convert.model_to_device(nn.Identity()), deepnet_bottom], TopModel(),
                               aggregator=lambda xs: xs,
                               modifiers=[Modifier(), get_modifier(sparsity)])
    train = SplitTrain(model, train_loader, test_loader, {"acc": acc, "auc": auc}, F.binary_cross_entropy,
                       lambda m: SGD(m, lr, momentum=0.9))

    record_name = gen_param_str(
        "criteo_wide-n-deep", {"modifier": modifier, "sparsity": sparsity, "optim": "sgd", "lr": lr})

    train.train(n_rounds, n_batches_per_round, record_name=record_name, save_model_n_round=(100, 100))


if __name__ == '__main__':
    save_dir = Path(f"Criteo-{seed}")
    if not save_dir.is_dir():
        save_dir.mkdir()
    os.chdir(save_dir)

    """ 
    !! Otherwise will deadlock, see https://github.com/pytorch/pytorch/issues/50669 
    Otherwise, write all global statements into function!!
    """

    # parallel_process([
    #     lambda: split_train_sparse(0.1, 0.01, "randTopk4", 1000, 200),
    #     lambda: split_train_sparse(0.05, 0.01, "randTopk4", 2000, 200),
    #     lambda: split_train_sparse(0.02, 0.01, "randTopk4", 2000, 200),
    # ], ["cuda:0"] * 3)
    #
    # parallel_process([
    #     lambda: split_train_sparse(0.1, 0.01, "randTopk2", 1000, 200),
    #     lambda: split_train_sparse(0.05, 0.01, "randTopk2", 2000, 200),
    #     lambda: split_train_sparse(0.02, 0.01, "randTopk2", 2000, 200),
    # ], ["cuda:0"] * 3)
    #
    # parallel_process([
    #     lambda: split_train_sparse(0.1, 0.01, "randTopk", 1000, 200),
    #     lambda: split_train_sparse(0.05, 0.01, "randTopk", 2000, 200),
    #     lambda: split_train_sparse(0.02, 0.01, "randTopk", 2000, 200),
    # ], ["cuda:0"] * 3)

    # parallel_process([
    #     lambda: split_train_sparse(0.1, 0.01, "random", 1000, 200),
    #     lambda: split_train_sparse(0.05, 0.01, "random", 2000, 200),
    #     lambda: split_train_sparse(0.02, 0.01, "random", 2000, 200),
    #     lambda: split_train_sparse(0.1, 0.01, "fixed", 1000, 200),
    #     lambda: split_train_sparse(0.05, 0.01, "fixed", 2000, 200),
    #     lambda: split_train_sparse(0.02, 0.01, "fixed", 2000, 200)
    # ], ["cuda:0"] * 3 + ["cuda:1"] * 3)

    parallel_process([
        lambda: centralized_train(0.01),
        lambda: only_wide_train(0.01),
        lambda: split_train_sparse(0.1, 0.01, "topk"),
        lambda: split_train_sparse(0.05, 0.01, "topk"),
        lambda: split_train_sparse(0.02, 0.01, "topk")
    ], ["cuda:1"] * 2 + ["cuda:0"] * 3)
