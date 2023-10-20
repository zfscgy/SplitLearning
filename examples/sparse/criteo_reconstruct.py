import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, SplitTrain, gen_param_str
from split_learn.modifiers import TopKModifier, RandomDropoutModifier, FixedSparseModifier, Modifier, RandomTopKModifier
from split_learn.nn.wide_and_deep import DeepNet, WideAndDeepNet

from deep_utils.convert import Convert
from deep_utils.datasets import CriteoSmall
from deep_utils.ops import clone_module, LambdaLayer
from deep_utils.multi_process_run import parallel_process


class CriteoRecover(Dataset):
    def __init__(self, original_dataset, n_samples: int):
        self.original_dataset = original_dataset
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x_n, x_c, y = self.original_dataset[index]
        return x_c, x_c[19].long()  # Only get "C33"


def get_reconstructor():
    return nn.Sequential(
        nn.Linear(256, 32), nn.Tanh(),
        nn.Linear(32, 4))


embedding_dim = 32
split_position = 1


def get_dnn_modules():
    return [
        nn.Sequential(nn.Linear(len(CriteoSmall.category_counts) * 32, 256), nn.ReLU()),
        nn.Sequential(nn.Linear(256, 32), nn.ReLU())
    ]


def get_model():
    return WideAndDeepNet(CriteoSmall.n_numerical_features, CriteoSmall.category_counts,
                          nn.Sequential(*get_dnn_modules()), 32, embedding_dim)


def acc(y_pred, y_true):
    return np.mean(np.argmax(y_pred, -1) == y_true)


def reconstruct(encoder_name: str, n_samples: int):
    train_data, test_data = CriteoSmall.get()
    train_data, test_data = CriteoRecover(train_data, n_samples), CriteoRecover(test_data, len(test_data))
    train_loader = DataLoader(train_data, 128)
    test_loader = DataLoader(test_data, 512)

    reconstruct_model = Convert.model_to_device(get_reconstructor())
    classification_model = get_model()
    classification_model.load_state_dict(torch.load("../" + encoder_name + "-model.pth"))
    classification_model = Convert.model_to_device(classification_model)

    classification_model.mlp.mlp = nn.Sequential(
        *list(classification_model.mlp.mlp.children())[:split_position])

    modules = [classification_model.mlp, reconstruct_model]
    model = nn.Sequential(*modules)

    for p in modules[0].parameters():
        p.requires_grad = False

    train = Train(model, train_loader, test_loader, {"acc": acc}, nn.CrossEntropyLoss(), Adam)
    train.train(500, 100, record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


def reconstruct_split(encoder_name: str, n_samples: int, sparsity: float, modifier: str = "topk"):
    train_data, test_data = CriteoSmall.get()
    train_data, test_data = CriteoRecover(train_data, n_samples), CriteoRecover(test_data, len(test_data))
    train_loader = DataLoader(train_data, 128)
    test_loader = DataLoader(test_data, 512)

    reconstruct_model = Convert.model_to_device(get_reconstructor())

    classification_model = Convert.model_to_device(
        WideAndDeepNet(
            CriteoSmall.n_numerical_features, CriteoSmall.category_counts,
            nn.Sequential(*get_dnn_modules()), 32, embedding_dim))


    split_position = 1
    deepnet_bottom = DeepNet(CriteoSmall.category_counts,
                             nn.Sequential(*list(classification_model.mlp.mlp.children())[:split_position]),
                             embedding_dim)
    deepnet_bottom = Convert.model_to_device(deepnet_bottom)

    class TopModel(nn.Module):
        def __init__(self):
            super(TopModel, self).__init__()
            self.dnn = nn.Sequential(*list(classification_model.mlp.mlp.children())[split_position:])
            self.prediction_model = classification_model.prediction_model

        def forward(self, x):
            x_wide, x_deep = x
            deep_out = self.dnn(x_deep)
            last_feature = torch.cat([deep_out, x_wide], dim=-1)
            return torch.sigmoid(self.prediction_model(last_feature))

    get_modifier = {
        "topk": TopKModifier,
        "random": RandomDropoutModifier,
        "fixed": lambda rate: FixedSparseModifier(rate, [256]),
        "randTopk": lambda rate: RandomTopKModifier(rate),
    }[modifier]
    split_model = TwoLayerSplitModel([Convert.model_to_device(nn.Identity()), deepnet_bottom], TopModel(),
                                     aggregator=lambda xs: xs,
                                     modifiers=[Modifier(), get_modifier(sparsity)])
    split_model.load("../" + encoder_name + "-model.pth")

    modules = [Convert.model_to_device(deepnet_bottom),
               LambdaLayer(split_model.modifiers[0].modify_forward_test),
               Convert.model_to_device(reconstruct_model)]
    model = nn.Sequential(*modules)
    for p in deepnet_bottom.parameters():
        p.requires_grad = False

    train = Train(model, train_loader, test_loader, {"acc": acc}, nn.CrossEntropyLoss(), Adam)
    train.train(500, 100, record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


if __name__ == '__main__':
    os.chdir("Criteo-20220307001")
    rec_dir = Path("Rec")
    if not rec_dir.is_dir():
        rec_dir.mkdir()
    os.chdir(rec_dir)
    prefix = "criteo_wide-n-deep-"
    # reconstruct("criteo_wide-n-deep-optim_adam")
    # reconstruct_split("criteo_wide-n-deep-modifier_topk_sparsity_0.1_lr_0.1_optim_sgd", 0.1)
    for i in [50, 100, 200, 400, 800]:
        parallel_process([
            lambda: reconstruct_split(prefix + "modifier_randTopk_sparsity_0.1_optim_sgd_lr_0.01", i, 0.1, "randTopk"),
            lambda: reconstruct_split(prefix + "modifier_randTopk_sparsity_0.05_optim_sgd_lr_0.01", i, 0.05, "randTopk"),
            lambda: reconstruct_split(prefix + "modifier_randTopk_sparsity_0.02_optim_sgd_lr_0.01", i, 0.02, "randTopk"),
            lambda: reconstruct_split(prefix + "modifier_topk_sparsity_0.1_optim_sgd_lr_0.01", i, 0.1, "topk"),
            lambda: reconstruct_split(prefix + "modifier_topk_sparsity_0.05_optim_sgd_lr_0.01", i,  0.05, "topk"),
            lambda: reconstruct_split(prefix + "modifier_topk_sparsity_0.02_optim_sgd_lr_0.01", i, 0.02, "topk"),
            # lambda: reconstruct(prefix + "optim_sgd_lr_0.01", i)
        ], ["cuda:0"] * 3 + ["cuda:1"] * 3)
