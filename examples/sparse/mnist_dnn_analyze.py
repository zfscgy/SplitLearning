import os
from pathlib import Path

import matplotlib.pyplot as plt
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
from split_learn.modifiers import FixedSparseModifier, RandomDropoutModifier, TopKModifier, RandomTopKModifier, Modifier
from split_learn.analyze import TopkReplacements, NeuronChangeMonitor, RoundMetricCallbacks, NeuronInteraction

from deep_utils.config import GlobalConfig
from deep_utils.ops import clone_module
from deep_utils.multi_process_run import parallel_process

train_set, test_set = Mnist.get(txs=[Mnist.tx_flatten])


def acc(y_pred, y_true):
    return np.mean(np.argmax(y_pred, -1) == y_true)


def centralized_sparsity(model_path: str):
    train_loader = DataLoader(train_set, 32)
    modules = [
        nn.Sequential(nn.Linear(784, 128), nn.ReLU()),
        nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
        nn.Linear(32, 10)
    ]

    model = nn.Sequential(*modules)
    model.load_state_dict(torch.load(model_path + ".pth"))


    head_model = nn.Sequential(*modules)
    head_model = Convert.model_to_device(head_model)
    for xs, _ in train_loader:
        y = head_model(Convert.to_tensor(xs))
        sparsity = Convert.to_numpy(torch.mean((y > 0).float()))
        print(f"Current batch sparsity: {sparsity:.4f}")


def analyze_distribution(modifier_name: str = "topk", sparsity: float = 0.1, lr: float = 0.01):
    train_set_raw, test_set_raw = Mnist.get(txs=[Mnist.tx_flatten])
    train_set = SplitXDataset(train_set_raw)
    test_set = SplitXDataset(test_set_raw)

    train_loader = DataLoader(train_set, 50)
    test_loader = DataLoader(test_set, 512)

    modifier = {
        "topk": TopKModifier(sparsity),
        "randTopk": RandomTopKModifier(sparsity),
        "randTopk2": RandomTopKModifier(sparsity, 2),
        "randTopk4": RandomTopKModifier(sparsity, 4),
    }[modifier_name]

    torch.manual_seed(seed)
    modules = [
        nn.Sequential(nn.Linear(784, 128), nn.Sigmoid()),
        nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
        nn.Linear(32, 10)
    ]

    model = TwoLayerSplitModel(
        [Convert.model_to_device(modules[0])],
         Convert.model_to_device(nn.Sequential(*modules[1:])),
        modifiers=[modifier]
    )
    train = SplitTrain(model, train_loader, test_loader, {"acc": acc}, nn.CrossEntropyLoss(),
                       lambda m: SGD(m, lr, momentum=0.9))

    record_name = gen_param_str(
        "analyze-mnist_dnn", {"modifier": modifier_name, "sparsity": sparsity, "optim": "sgd", "lr": lr})

    mask_change_counter = TopkReplacements(model, sparsity, train_loader, 50)
    neuron_change_monitor = NeuronChangeMonitor(model, sparsity, train_loader, 50)

    roundcaller = RoundMetricCallbacks(record_name + "_round-metrics.csv", {
        "maskChange": (lambda _: mask_change_counter.count()),
        "largeNeuronChange": (lambda _: neuron_change_monitor.get_large_change()),
        "smallNeuronChange": (lambda _: neuron_change_monitor.get_small_change())
    })

    train.train(100, 1000, round_callback=roundcaller, record_name=record_name, save_model_n_round=(100, 100))


def analyze_neuron_interaction(lr: float=0.01, hidden_size: int=20):
    train_set_raw, test_set_raw = Mnist.get(txs=[Mnist.tx_flatten])
    train_set = SplitXDataset(train_set_raw)
    test_set = SplitXDataset(test_set_raw)

    train_loader = DataLoader(train_set, 50)
    test_loader = DataLoader(test_set, 512)

    torch.manual_seed(seed)
    modules = [
        nn.Sequential(nn.Linear(784, hidden_size), nn.ReLU()),
        nn.Sequential(nn.Linear(hidden_size, 32), nn.Tanh()),
        nn.Linear(32, 10)
    ]

    bottom_model = Convert.model_to_device(nn.Sequential(*modules[:2]))

    model = TwoLayerSplitModel(
        [bottom_model],
         Convert.model_to_device(nn.Sequential(*modules[2:])),
        modifiers=[Modifier()]
    )
    train = SplitTrain(model, train_loader, test_loader, {"acc": acc}, nn.CrossEntropyLoss(),
                       lambda m: SGD(m, lr, momentum=0.9))


    record_name = gen_param_str(
        "analyze-mnist_dnn-interaction", {"optim": "sgd", "lr": lr, "hsize": hidden_size})

    neuron_interaction = NeuronInteraction(bottom_model, train_set_raw)

    metric_callbacks = RoundMetricCallbacks(record_name + "-round_metrics", {
        "inSelf": lambda _: neuron_interaction.get_in_sample_self(),
        "inCross": lambda _: neuron_interaction.get_in_sample_cross(),
        "crossSelf": lambda _: neuron_interaction.get_cross_sample_self(),
        "crossCross": lambda _: neuron_interaction.get_cross_sample_cross(),
        "inSelf2": lambda _: neuron_interaction.get_other_stats(0),
        "inSelf01": lambda _: neuron_interaction.get_other_stats(1),
        "inCross01": lambda _: neuron_interaction.get_other_stats(2),
    })

    train.train(100, 100, round_callback=metric_callbacks, record_name=record_name, save_model_n_round=(100, 100))



if __name__ == '__main__':
    GlobalConfig.device = "cuda:1"
    seed = 20220307001
    save_dir = Path(f"Mnist-{seed}")
    os.chdir(save_dir)
#    centralized_sparsity("Mnist-20220307001/mnist_dnn-optim_adam-model")
#    analyze_neuron_interaction(0.0, 1)
    lr = 0.0001
    parallel_process([
        lambda: analyze_neuron_interaction(lr, 50),
        lambda: analyze_neuron_interaction(lr, 100),
        lambda: analyze_neuron_interaction(lr, 150),
        lambda: analyze_neuron_interaction(lr, 200),
        lambda: analyze_neuron_interaction(lr, 250),
        lambda: analyze_neuron_interaction(lr, 300),
    ], ["cuda:0"] * 3 + ["cuda:1"] * 3)


