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
from split_learn.modifiers import Modifier, TopKModifier, RandomTopKModifier
from split_learn.nn.resnet import make_resnet20_modules

from deep_utils.convert import Convert
from deep_utils.datasets import Cifar10
from deep_utils.ops import clone_module, LambdaLayer
from deep_utils.multi_process_run import parallel_process


class CifarReconstructionData(Dataset):
    def __init__(self, original_data, n_samples: int):
        self.original_data = original_data
        self.n_samples = n_samples

    def __getitem__(self, index: int):
        x, y = self.original_data[index]
        return x, x

    def __len__(self):
        return self.n_samples


cifar_train, cifar_test = Cifar10.get()


def l1_dist(pred_ys: np.ndarray, ys: np.ndarray):
    return np.mean(np.abs(pred_ys - ys))


def get_decoder():
    return nn.Sequential(
        nn.Conv2d(32, 16, (3, 3), padding=(1, 1)),
        nn.LeakyReLU(),
        nn.Conv2d(16, 8, (3, 3), padding=(1, 1)),
        nn.LeakyReLU(),
        nn.Conv2d(8, 3, (1, 1)),
        nn.Sigmoid()
    )


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
        image_raw = np.moveaxis(Convert.to_numpy(test_images[i]), [0, 1, 2], [2, 0, 1])
        image_reconstructed = np.moveaxis(Convert.to_numpy(reconstructed_images[i]), [0, 1, 2], [2, 0, 1])
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
    cifar_reconstruction_train = CifarReconstructionData(cifar_train, n_samples)
    cifar_reconstruction_test = CifarReconstructionData(cifar_test, len(cifar_test))
    train_loader = DataLoader(cifar_reconstruction_train, 32)
    test_loader = DataLoader(cifar_reconstruction_test, 128)

    resnet20_modules = make_resnet20_modules()
    resnet20_model = nn.Sequential(*resnet20_modules)

    model = Convert.model_to_device(
        nn.Sequential(
            *resnet20_modules[:2],
            get_decoder()
    ))

    resnet20_model.load_state_dict(torch.load("../" + encoder_name + "-model.pth"))
    for p in resnet20_modules[0].parameters():
        p.requires_grad = False
    for p in resnet20_modules[1].parameters():
        p.requires_grad = False

    train = Train(model, train_loader, test_loader, {"L1": l1_dist}, nn.L1Loss(), Adam)
    train.train(1000, 100, round_callback=plot_image,
                record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


def train_reconstruction_split(encoder_name: str, n_samples: int, modifier_name: str, sparsity: float):
    cifar_reconstruction_train = CifarReconstructionData(cifar_train, n_samples)
    cifar_reconstruction_test = CifarReconstructionData(cifar_test, len(cifar_test))
    train_loader = DataLoader(cifar_reconstruction_train, 32)
    test_loader = DataLoader(cifar_reconstruction_test, 128)

    resnet20_modules = make_resnet20_modules()

    modifer = {
        "topk": TopKModifier(sparsity),
        "randTopk": RandomTopKModifier(sparsity)
    }[modifier_name]

    split_model = TwoLayerSplitModel(
        [Convert.model_to_device(nn.Sequential(*resnet20_modules[:2]))],
         Convert.model_to_device(nn.Sequential(*resnet20_modules[2:])),
        modifiers=[modifer])

    split_model.load("../" + encoder_name + "-model.pth")
    decoder = Convert.model_to_device(get_decoder())
    model = nn.Sequential(
        split_model.bottom_models[0],
        LambdaLayer(split_model.modifiers[0].modify_forward_test),
        decoder)
    for p in split_model.bottom_models[0].parameters():
        p.requires_grad = False

    train = Train(model, train_loader, test_loader, {"L1": l1_dist}, nn.L1Loss(), Adam)
    train.train(1000, 100, round_callback=plot_image,
                record_name=gen_param_str(encoder_name + "-rec", {"samples": n_samples}))


if __name__ == '__main__':
    os.chdir("Cifar10-20220307001")
    rec_dir = Path("Rec")
    if not rec_dir.is_dir():
        rec_dir.mkdir()
    os.chdir(rec_dir)
    prefix = "cifar_resnet-"
    for i in [100, 200, 400, 800]:
        parallel_process([
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.1_optim_sgd_lr_0.01", i, "randTopk",
                                               0.1),
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.05_optim_sgd_lr_0.01", i, "randTopk",
                                               0.05),
            lambda: train_reconstruction_split(prefix + "modifier_randTopk_sparsity_0.02_optim_sgd_lr_0.01", i, "randTopk",
                                               0.02),
            # lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.1_optim_sgd_lr_0.01", i, "topk",
            #                                    0.1),
            # lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.05_optim_sgd_lr_0.01", i, "topk",
            #                                    0.05),
            # lambda: train_reconstruction_split(prefix + "modifier_topk_sparsity_0.02_optim_sgd_lr_0.01", i, "topk",
            #                                    0.02),
            # lambda: train_reconstruction(prefix + "optim_sgd_lr_0.01", i)
        ], ["cuda:0"] * 3)

    # parallel_process([
    #     lambda: train_reconstruction(prefix + "optim_sgd_lr_0.01", 50),
    #     lambda: train_reconstruction(prefix + "optim_sgd_lr_0.01", 100),
    #     lambda: train_reconstruction(prefix + "optim_sgd_lr_0.01", 200),
    #     lambda: train_reconstruction(prefix + "optim_sgd_lr_0.01", 400),
    #     lambda: train_reconstruction(prefix + "optim_sgd_lr_0.01", 800),
    # ], ["cuda:0"] * 5)
