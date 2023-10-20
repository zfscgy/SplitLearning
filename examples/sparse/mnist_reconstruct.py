import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from split_learn.models import TwoLayerSplitModel
from split_learn.train import Train, gen_param_str
from split_learn.modifiers import TopKModifier, RandomTopKModifier

from deep_utils.convert import Convert
from deep_utils.datasets import Mnist
from deep_utils.ops import clone_module, LambdaLayer
from deep_utils.multi_process_run import parallel_process


class MnistReconstructionData(Dataset):
    def __init__(self, original_data, n_samples: int):
        self.original_data = original_data
        self.n_samples = n_samples

    def __getitem__(self, index: int):
        x, y = self.original_data[index]
        return x, x

    def __len__(self):
        return self.n_samples


mnist_train, mnist_test = Mnist.get(txs=[Mnist.tx_flatten])


def l1_dist(pred_ys: np.ndarray, ys: np.ndarray):
    return np.mean(np.abs(pred_ys - ys))


def get_decoder():
    return nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 784),
            nn.Sigmoid())


initial_decoder = get_decoder()


def plot_image(train: Train):
    if train.current_round == 0 or train.current_round % 200 != 0:
        return

    dir = Path(train.record_name + "-Imgs")
    if not dir.is_dir():
        dir.mkdir()

    test_images, _ = next(iter(train.validation_loader))
    # Extract 10 images
    test_images = test_images[:10]
    reconstructed_images = train.model(Convert.to_tensor(test_images))

    figs = []
    for i in range(10):
        image_raw = Convert.to_numpy(test_images[i].view(28, 28))
        image_reconstructed = Convert.to_numpy(reconstructed_images[i].view(28, 28))
        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image_raw)
        ax1.set_axis_off()
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(image_reconstructed)
        ax2.set_axis_off()
        fig.show()
        figs.append(fig)
        fig.savefig(dir.joinpath(f"image_{i}-round_{train.current_round}.png"))


def train_reconstruction(encoder_name: str, n_samples: int):
    mnist_reconstruction_train = MnistReconstructionData(mnist_train, n_samples)
    mnist_reconstruction_test = MnistReconstructionData(mnist_test, len(mnist_test))
    train_loader = DataLoader(mnist_reconstruction_train, 32)
    test_loader = DataLoader(mnist_reconstruction_test, 128)

    dnn_modules = [
        nn.Sequential(nn.Linear(784, 128), nn.ReLU()),
        nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
        nn.Linear(32, 10)
    ]

    classification_model = nn.Sequential(*dnn_modules)
    model = Convert.model_to_device(
        nn.Sequential(
            dnn_modules[0],
            get_decoder()
    ))
    classification_model.load_state_dict(torch.load("../" + encoder_name + "-model.pth"))
    for p in dnn_modules[0].parameters():
        p.requires_grad = False

    train = Train(model, train_loader, test_loader, {"L1": l1_dist}, nn.L1Loss(), Adam)
    train.train(1000, 100, round_callback=plot_image,
                record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


def train_reconstruction_split(encoder_name: str, n_samples: int, modifier_name: str, sparsity: float):
    mnist_reconstruction_train = MnistReconstructionData(mnist_train, n_samples)
    mnist_reconstruction_test = MnistReconstructionData(mnist_test, len(mnist_test))
    train_loader = DataLoader(mnist_reconstruction_train, 32)
    test_loader = DataLoader(mnist_reconstruction_test, 128)

    modules = [
        nn.Sequential(nn.Linear(784, 128), nn.ReLU()),
        nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
        nn.Linear(32, 10)
    ]

    modifier = {
        "topk": TopKModifier(sparsity),
        "randTopk": RandomTopKModifier(sparsity)
    }[modifier_name]

    split_model = TwoLayerSplitModel(
        [Convert.model_to_device(modules[0])],
         Convert.model_to_device(nn.Sequential(*modules[1:])),
        modifiers=[modifier])
    split_model.load("../" + encoder_name + "-model.pth")
    decoder = Convert.model_to_device(get_decoder())
    model = nn.Sequential(
        split_model.bottom_models[0],
        LambdaLayer(split_model.modifiers[0].modify_forward_test),
        decoder)
    for p in split_model.bottom_models[0].parameters():
        p.requires_grad = False

    train = Train(model, train_loader, test_loader, {"L1": l1_dist}, nn.L1Loss(), Adam)
    train.train(200, 100, round_callback=plot_image,
                record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


if __name__ == '__main__':
    os.chdir("Mnist-20220307001")
    rec_dir = Path("Rec")
    if not rec_dir.is_dir():
        rec_dir.mkdir()
    os.chdir(rec_dir)
    prefix = "mnist_dnn-"
    for i in [50, 100, 200, 400, 800]:
        parallel_process([
            # lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.1_optim_sgd_lr_0.01", i, "topk",
            #                                    0.1),
            # lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.05_optim_sgd_lr_0.01", i, "topk",
            #                                    0.05),
            # lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.02_optim_sgd_lr_0.01", i, "topk",
            #                                    0.02),
            # lambda: train_reconstruction(prefix + "optim_sgd_lr_0.01", i),
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.1_optim_sgd_lr_0.01", i, "topk",
                                               0.1),
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.05_optim_sgd_lr_0.01", i, "topk",
                                               0.05),
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.02_optim_sgd_lr_0.01", i, "topk",
                                               0.02),
        ], ["cuda:0"] * 3)
