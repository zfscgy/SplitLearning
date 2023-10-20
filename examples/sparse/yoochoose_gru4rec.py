import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset

from deep_utils.config import GlobalConfig
from deep_utils.datasets import YooChoose64
from deep_utils.convert import Convert
from deep_utils.ops import clone_module
from deep_utils.multi_process_run import parallel_process

from split_learn.data import SplitXDataset
from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, gen_param_str, SplitTrain
from split_learn.nn.gru4rec import make_gru4rec_modules, bpr_loss
from split_learn.modifiers import Modifier, FixedSparseModifier, RandomDropoutModifier, TopKModifier, RandomTopKModifier


batch_size = 32

input_len = 12
embedding_dim = 32
gru_size = 300
output_hidden_size = 32





def hr_20(pred_ys: np.ndarray, ys: np.ndarray):
    """
    Hit ratio @ 20
    :param pred_ys:
    :param ys:
    :return:
    """
    top20_indices = np.argpartition(-pred_ys, 20)[:, :20]
    return np.mean(np.sum(top20_indices == ys[:, np.newaxis], axis=-1))


def centralized_train(lr: float, n_rounds: int = 500, n_batches_per_round: int = 10000):
    train_set, test_set = YooChoose64.get(input_len)
    train_loader, test_loader = DataLoader(train_set, batch_size), DataLoader(test_set, 256)

    gru4rec_modules = make_gru4rec_modules(YooChoose64.n_items, embedding_dim, gru_size, output_hidden_size)
    gru4rec = Convert.model_to_device(nn.Sequential(*gru4rec_modules))
    train = Train(gru4rec, train_loader, test_loader, {"acc": hr_20}, bpr_loss, lambda ps: SGD(ps, lr, 0.9))
    record_name = gen_param_str("YooChoose64_gru4rec", {"optim": "sgd", "lr": lr})
    train.train(n_rounds, n_batches_per_round, record_name=record_name, save_model_n_round=(100, 100))


def split_train(lr: float, sparsity: float, modifier: str, n_rounds: int = 500, n_batches_per_round: int = 10000):
    train_set, test_set = YooChoose64.get(input_len)
    train_set_split, test_set_split = SplitXDataset(train_set), SplitXDataset(test_set)
    train_loader_split, test_loader_split = DataLoader(train_set_split, batch_size), DataLoader(test_set_split, 256)

    gru4rec_modules = make_gru4rec_modules(YooChoose64.n_items, embedding_dim, gru_size, output_hidden_size)
    bottom_model = Convert.model_to_device(nn.Sequential(*gru4rec_modules[:2]))
    top_model = Convert.model_to_device(nn.Sequential(*gru4rec_modules[2:]))

    get_modifier = {
        "topk": TopKModifier,
        "randTopk": RandomTopKModifier,
        "randTopk2": lambda rate: RandomTopKModifier(rate, 2),
        "randTopk4": lambda rate: RandomTopKModifier(rate, 4),
        "random": RandomDropoutModifier,
        "fixed": lambda rate: FixedSparseModifier(rate, [gru_size])
    }[modifier]

    model = TwoLayerSplitModel([bottom_model], top_model, modifiers=[get_modifier(sparsity)])
    train = SplitTrain(model, train_loader_split, test_loader_split, {"acc": hr_20}, bpr_loss,
                       lambda m: SGD(m, lr, momentum=0.9))

    record_name = gen_param_str("YooChoose64_gru4rec",
                                {"modifier": modifier, "sparsity": sparsity, "optim": "sgd", "lr": lr})

    train.train(n_rounds, n_batches_per_round, record_name=record_name, save_model_n_round=(100, 100))


if __name__ == '__main__':
    seed = 20220307001
    torch.manual_seed(seed)
    save_dir = Path(f"YooChoose64-{seed}")
    if not save_dir.is_dir():
        save_dir.mkdir()
    os.chdir(save_dir)
    # centralized_train(0.05, 1000, 2000)
    parallel_process([
        lambda: split_train(0.01, 0.1, "topk", 1000, 3000),
    ], ["cuda:1"])

    # parallel_process([
    #     lambda: split_train(0.05, 0.1, "topk", 1000, 3000),
    #     lambda: split_train(0.05, 0.05, "topk", 1000, 4000),
    #     lambda: split_train(0.05, 0.02, "topk", 1000, 4000),
    #     lambda: split_train(0.05, 0.1, "randTopk2", 1000, 3000),
    #     lambda: split_train(0.05, 0.05, "randTopk2", 1000, 4000),
    #     lambda: split_train(0.05, 0.02, "randTopk2", 1000, 4000),
    #     lambda: split_train(0.05, 0.1, "randTopk", 1000, 3000),
    #     lambda: split_train(0.05, 0.05, "randTopk", 1000, 4000),
    #     lambda: split_train(0.05, 0.02, "randTopk", 1000, 4000),
    #     lambda: centralized_train(0.05, 1000, 2000)
    # ], ["cuda:0"] * 5 + ["cuda:1"] * 5)
    #
    #
    # parallel_process([
    #     lambda: split_train(0.05, 0.1, "randTopk4", 1000, 3000),
    #     lambda: split_train(0.05, 0.05, "randTopk4", 1000, 4000),
    #     lambda: split_train(0.05, 0.02, "randTopk4", 1000, 4000),
    #     lambda: split_train(0.05, 0.1, "fixed", 1000, 3000),
    #     lambda: split_train(0.05, 0.05, "fixed", 1000, 4000),
    #     lambda: split_train(0.05, 0.02, "fixed", 1000, 4000),
    #     lambda: split_train(0.05, 0.1, "random", 1000, 3000),
    #     lambda: split_train(0.05, 0.05, "random", 1000, 4000),
    #     lambda: split_train(0.05, 0.02, "random", 1000, 4000),
    # ],  ["cuda:0"] * 4 + ["cuda:1"] * 5)
