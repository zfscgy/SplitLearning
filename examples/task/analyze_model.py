import argparse
from typing import Callable, List

import PIL.ImageDraw2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from functools import partial

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from deep_utils.datasets import Cifar100
from deep_utils.convert import Convert


from split_learn.models import Resnet20Factory
from split_learn.modifiers import TopKModifier, Modifier
from split_learn.train import run_single_label_attack_task, run_single_input_attack_task, \
    run_single_generation_attack_cifar

parser = argparse.ArgumentParser("Input index")
parser.add_argument("-i", nargs=1, type=int, dest="index")
args = parser.parse_args()
if args.index is not None:
    index = args.index[0]
else:
    index = 0
print("Index:", index)


# cifar100_train, cifar100_test = Cifar100.get(txs_train=Cifar100.get_txs_random_transform())
cifar100_train, cifar100_test = Cifar100.get()
cifar_config = {
    "model_factory": Resnet20Factory(100),
    "sparsity": 0.02,
    "split_layer": 5,
    "train_set": cifar100_train,
    "test_set": cifar100_test
}


def analyze_topk_distribution(conf: dict, model_paths: List[str], titles: List[str],
                              export_file: str = None, extra_plt_cmds: Callable=None):
    topk_modifier = TopKModifier(conf['sparsity'], conf['model_factory'].get_splits()[conf['split_layer'] - 1])

    def get_hist(model_path: str):
        # return np.random.uniform(0, 4000, [128])
        model = conf['model_factory'].get_modified_model(conf['split_layer'], topk_modifier)
        model.load_state_dict(torch.load(model_path))
        Convert.model_to_device(model)
        bottom_model = conf['model_factory'].get_bottom_model()

        train_set = conf['train_set']
        data_loader = DataLoader(train_set, batch_size=256)

        ind_counts = Convert.to_tensor(torch.zeros(*topk_modifier.input_shape))

        bottom_model.eval()
        with torch.no_grad():
            for xs, _ in data_loader:
                topk_modifier(bottom_model(Convert.to_tensor(xs)))
                ind_counts += topk_modifier.mask.sum(dim=0)

        ind_counts = Convert.to_numpy(ind_counts)
        return ind_counts

    ind_counts_all = [get_hist(model_path) for model_path in model_paths]

    plt.figure(figsize=(23, 6))
    plt.tight_layout()
    subplots = [141, 142, 143, 144]
    for i in range(4):
        plt.subplot(subplots[i])
        plt.hist(ind_counts_all[i], bins=20, range=[0, 4000])
        plt.xlim(0, 4000)
        plt.xticks([0, 2000, 4000], fontsize=28)
        if i == 0:
            plt.ylim(0, 40)
            plt.yticks([0, 20, 40], fontsize=28)
        else:
            plt.yticks([])
        plt.title(titles[i], y=1, fontsize=28)

    plt.gcf().text(0.5, -0.03, r"#Time being top-$k$", ha="center", va="center",
                   fontsize=32, fontfamily="sans-serif")
    plt.gcf().text(0.07, 0.5, "#Neurons", ha="center", va="center", rotation="vertical",
                   fontsize=32, fontfamily="sans-serif")

    if extra_plt_cmds is not None:
        extra_plt_cmds()
    if export_file is not None:
        plt.savefig(export_file, bbox_inches='tight')
    plt.show()


def concatenate_images(folders: List[str], image_size: int = 32, n_examples: int = 10, names: List[str] = None,
                       export_file: str = None):
    stacked_img = np.zeros([(1 + len(folders)) * image_size, n_examples * image_size, 4])
    for i, folder in enumerate(folders):
        if i == 0:
            for j in range(n_examples):
                stacked_img[0: image_size, j * image_size: (j + 1) * image_size, :] =\
                    np.array(Image.open(f'{folder}/{j}-origin.png'))

        for j in range(n_examples):
            stacked_img[(i + 1) * image_size: (i + 2) * image_size, j * image_size: (j + 1) * image_size, :] = \
                np.asarray(Image.open(f'{folder}/{j}-regen.png'))

    if names is not None:
        resize_scale = 8
        font = ImageFont.truetype("arial.ttf", 10 * resize_scale)
        stacked_img = np.concatenate([np.zeros([(1 + len(folders)) * image_size, int(2.5 * image_size), 4]), stacked_img], axis=1)
        stacked_img = Image.fromarray(stacked_img.astype(np.uint8))
        stacked_img = stacked_img.resize([stacked_img.size[0] * resize_scale, stacked_img.size[1] * resize_scale])
        draw = ImageDraw.ImageDraw(stacked_img)

        for i, label in enumerate(["Origin"] + names):
            draw.text((8 * resize_scale, int((i + 0.3) * 32 * resize_scale)), label, 'black', font)

        stacked_img = np.array(stacked_img).astype(np.uint8)
        plt.imshow(stacked_img)
        if export_file is not None:
            plt.imsave(export_file, stacked_img)
        plt.show()

if __name__ == '__main__':
    analyze_topk_distribution(
        cifar_config,
        ["analyze-model/cifar100-splitLayer_5_modifier_topk-2.8564_seed_1926.pth",
         "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.05-2.8564_seed_1926.pth",
         "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.1-2.8564_seed_1926.pth",
         "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.2-2.8564_seed_1926.pth"],
        [r"$\alpha=0$ (top-$k$)", r"$\alpha=0.05$", r"$\alpha=0.1$", r"$\alpha=0.2$"],
        export_file="plots/cifar100-dist-topk.pdf")
    # analyze_topk_distribution(cifar_config,
    #                           "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.05-2.8564_seed_1926.pth",
    #                           export_file="plots/cifar100-dist-randtopk0.05.pdf")
    # analyze_topk_distribution(cifar_config,
    #                           "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.05-2.8564_seed_1926.pth",
    #                           export_file="plots/cifar100-dist-randtopk0.1.pdf")
    # analyze_topk_distribution(cifar_config,
    #                           "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.2-2.8564_seed_1926.pth",
    #                           export_file="plots/cifar100-dist-randtopk0.2.pdf")

    # paras = list(zip(
    #     ["analyze-model/cifar100-splitLayer_5_modifier_identity-100.0000_seed_1926.pth",
    #      "analyze-model/cifar100-splitLayer_5_modifier_topk-2.8564_seed_1926.pth",
    #      "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.05-2.8564_seed_1926.pth",
    #      "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.1-2.8564_seed_1926.pth",
    #      "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.2-2.8564_seed_1926.pth"],
    #     [Modifier(),
    #      TopKModifier(0.02, [128]),
    #      TopKModifier(0.02, [128]),
    #      TopKModifier(0.02, [128]),
    #      TopKModifier(0.02, [128])]
    # ))
    # model_path, modifier = paras[index]
    # run_single_label_attack_task(
    #     cifar_config['model_factory'],
    #     model_path,
    #     partial(TopKModifier, 0.02),
    #     cifar_config['split_layer'],
    #     cifar_config['train_set']
    # )
    # run_single_input_attack_task(
    #     cifar_config['model_factory'],
    #     model_path,
    #     partial(TopKModifier, 0.02),
    #     cifar_config['split_layer'],
    #     cifar_config['train_set']
    # )

    # run_single_generation_attack_cifar(
    #     cifar_config['model_factory'],
    #     model_path,
    #     partial(TopKModifier, 0.02),
    #     cifar_config['split_layer'],
    #     cifar_config['train_set'],
    #     cifar_config['test_set']
    # )

    # concatenate_images([
    #     "analyze-model/cifar100-splitLayer_5_modifier_identity-100.0000_seed_1926-gen",
    #     "analyze-model/cifar100-splitLayer_5_modifier_topk-2.8564_seed_1926-gen",
    #     "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.05-2.8564_seed_1926-gen",
    #     "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.1-2.8564_seed_1926-gen",
    #     "analyze-model/cifar100-splitLayer_5_modifier_nRandTopk0.2-2.8564_seed_1926-gen",
    #
    # ], names=["Non-sparse", "Topk", "RandTopk-0.05", "RandTopk-0.1", "RandTopk-0.2"],
    # export_file="plots/inversion-attack.pdf")

"""
export CUDA_VISIBLE_DEVICES=0
nohup python analyze_model.py -i 0 > a0.log &
nohup python analyze_model.py -i 1 > a1.log &
export CUDA_VISIBLE_DEVICES=1
nohup python analyze_model.py -i 2 > a2.log &
nohup python analyze_model.py -i 3 > a3.log &
nohup python analyze_model.py -i 4 > a4.log &
"""
