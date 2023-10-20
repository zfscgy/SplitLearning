import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset

from deep_utils.datasets import YooChoose64
from deep_utils.convert import Convert
from deep_utils.ops import clone_module, LambdaLayer
from deep_utils.multi_process_run import parallel_process
from deep_utils.config import GlobalConfig

from split_learn.data import SplitXDataset
from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, gen_param_str, SplitTrain
from split_learn.nn.gru4rec import make_gru4rec_modules, bpr_loss
from split_learn.modifiers import Modifier, FixedSparseModifier, RandomDropoutModifier, TopKModifier, RandomTopKModifier

class YoochooseReconstructionData(Dataset):
    def __init__(self, original_seq, original_data, n_samples: int):
        self.contain_first_150 = []
        for session in original_seq:
            session = np.array(session)
            self.contain_first_150.append(np.sum((session > 0) * (session < 150)) > 0)
        self.contain_first_150 = torch.tensor(self.contain_first_150).view(-1, 1).float()
        self.original_data = original_data
        self.n_samples = n_samples

    def __getitem__(self, index: int):
        x, _ = self.original_data[index]
        return x, self.contain_first_150[index]

    def __len__(self):
        return self.n_samples


gru_size = 300
input_len = 12
embedding_dim = 32
output_hidden_size = 32


def acc(ys_pred, ys_true):
    return np.mean(np.round(ys_pred) == ys_true)


def get_decoder():
    return nn.Sequential(nn.Linear(300, 128), nn.Tanh(), nn.Linear(128, 1), nn.Sigmoid())


def train_reconstruction(encoder_name: str, n_samples: int):
    train_set, test_set = YooChoose64.get(input_len)
    reconstruction_train_set = YoochooseReconstructionData(YooChoose64.original_seq_train, train_set, n_samples)
    reconstruction_test_set = YoochooseReconstructionData(YooChoose64.original_seq_train, test_set, n_samples)

    decoder = get_decoder()

    modules = make_gru4rec_modules(YooChoose64.n_items, embedding_dim, gru_size, output_hidden_size)
    learned_model = nn.Sequential(*modules)
    learned_model.load_state_dict(torch.load("../" + encoder_name + "-model.pth"))

    for p in modules[0].parameters():
        p.requires_grad = False
    for p in modules[1].parameters():
        p.requires_grad = False

    model = Convert.model_to_device(
        nn.Sequential(
            *modules[:2],
            decoder
    ))

    train = Train(model, DataLoader(reconstruction_train_set, 32), DataLoader(reconstruction_test_set, 256),
                  {"acc": acc}, F.binary_cross_entropy, Adam)
    train.train(500, 100, record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


def train_reconstruction_split(encoder_name: str, n_samples: int, modifier_name: str, sparsity: float):
    train_set, test_set = YooChoose64.get(input_len)
    reconstruction_train_set = YoochooseReconstructionData(YooChoose64.original_seq_train, train_set, n_samples)
    reconstruction_test_set = YoochooseReconstructionData(YooChoose64.original_seq_test, test_set, len(test_set))

    decoder = get_decoder()

    modules = make_gru4rec_modules(YooChoose64.n_items, embedding_dim, gru_size, output_hidden_size)

    modifer = {
        "topk": TopKModifier(sparsity),
        "randTopk": RandomTopKModifier(sparsity)
    }[modifier_name]

    learned_split_model = TwoLayerSplitModel([nn.Sequential(*modules[:2])], nn.Sequential(*modules[2:]),
                                             modifiers=[modifer])

    learned_split_model.load("../" + encoder_name + "-model.pth")

    for p in learned_split_model.bottom_models[0].parameters():
        p.requires_grad = False

    model = nn.Sequential(learned_split_model.bottom_models[0],
                          LambdaLayer(learned_split_model.modifiers[0].modify_forward_test),
                          decoder)
    model = Convert.model_to_device(model)
    train = Train(model, DataLoader(reconstruction_train_set, 32), DataLoader(reconstruction_test_set, 256),
                  {"acc": acc}, F.binary_cross_entropy, Adam)
    train.train(500, 100, record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


if __name__ == '__main__':
    os.chdir("YooChoose64-20220307001")
    rec_dir = Path("Rec")
    if not rec_dir.is_dir():
        rec_dir.mkdir()
    os.chdir(rec_dir)
    prefix = "YooChoose64_gru4rec-"
    for i in [100, 200, 400, 800, 1600]:
        parallel_process([
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.02_optim_sgd_lr_0.05", i,
                                               "randTopk", 0.02),
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.05_optim_sgd_lr_0.05", i,
                                               "randTopk", 0.05),
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.1_optim_sgd_lr_0.05", i,
                                               "randTopk", 0.1),
            lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.02_optim_sgd_lr_0.05", i,
                                               "topk", 0.02),
            lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.05_optim_sgd_lr_0.05", i,
                                               "topk", 0.05),
            lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.1_optim_sgd_lr_0.05", i,
                                               "topk", 0.1),
            lambda: train_reconstruction(prefix + "optim_sgd_lr_0.05", i)
        ], ["cuda:0"] * 5 + ["cuda:1"] * 2)