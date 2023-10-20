import os
from typing import Any, Callable, Dict, List, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader, Dataset



from deep_utils.train import Train, gen_param_str
from deep_utils.convert import Convert, GlobalConfig
from deep_utils.metrics import Metric, NegL2Loss
from deep_utils.ops import ViewLayer
from deep_utils.data import IdentityDataset

from split_learn.modifiers import Modifier
from split_learn.models import SplitModelFactory

import matplotlib.pyplot as plt


class SplitTask:
    def __init__(self, model_factory: SplitModelFactory,
                 get_modifier: Callable[[List[int]], Modifier], split_layer: int,
                 train_loader: DataLoader, validation_loader: DataLoader, test_loader: DataLoader,
                 loss_func: Callable, get_optimizer: Callable[[Any], Optimizer],
                 metrics: List[Metric], callback_metrics: Dict[str, Callable] = None,
                 seed: int = None,
                 n_rounds: int = None, batches_per_round: int = None, round_callback: Callable = None,
                 early_stop: int = 100, save_model: bool = False, prefix: str = "SplitTask"
                 ):
        self.model_factory = model_factory
        self.bottom_shape = model_factory.get_splits()[split_layer - 1]
        self.modifier = get_modifier(self.bottom_shape)

        def make_model():
            return model_factory.get_modified_model(split_layer, self.modifier, seed)

        self.make_model = make_model
        self.record_name = gen_param_str(
            prefix, {
                "splitLayer": split_layer,
                "modifier": self.modifier.name + f"-{100*self.modifier.compression_rate:.4f}",
                "seed": seed
            })

        self.train = Train(
            make_model, train_loader, validation_loader, test_loader,
            loss_func, get_optimizer, metrics, callback_metrics, n_rounds, batches_per_round, round_callback,
            early_stop, save_model=save_model, record_name=self.record_name
        )

    def run(self):
        if Path(self.record_name + ".csv").is_file():
            print(f"Task {self.record_name} already finished, return...")
            return
        self.train.train()


def run_single_label_attack_task(
        model_factory: SplitModelFactory,
        model_path: str,
        get_modifier: Callable[[List[int]], Modifier], split_layer: int,
        dataset: Dataset,
        n_test_samples: int = 100
):
    test_sample_batch = [Convert.to_tensor(dataset[i][0]) for i in range(n_test_samples)]
    test_sample_batch = torch.stack(test_sample_batch, dim=0)
    test_label_batch = [dataset[i][1] for i in range(n_test_samples)]

    modifier = get_modifier(model_factory.get_splits()[split_layer - 1])
    split_model = model_factory.get_modified_model(split_layer, modifier)
    split_model.load_state_dict(torch.load(model_path))
    Convert.model_to_device(split_model)

    bottom_model = model_factory.get_bottom_model()

    with torch.no_grad():
        test_sample_batch_features = modifier(bottom_model(test_sample_batch))
        # [N, dim]

    test_sample_batch_features = test_sample_batch_features[:, None, :]

    labels = []
    inner_products = []

    with torch.no_grad():
        for xs, ys in DataLoader(dataset, 256):
            labels.extend(Convert.to_numpy(ys).tolist())
            features = modifier(bottom_model(Convert.to_tensor(xs)))
            inner_products.append(torch.sum(test_sample_batch_features * features[None, :, :], dim=-1))
            # [N, batch]

    inner_products = torch.cat(inner_products, dim=1)

    aucs = []
    for i in range(n_test_samples):
        is_same_class = (np.array(labels) == test_label_batch[i]).astype(np.float)
        inner_product_scores = Convert.to_numpy(inner_products[i])
        auc_score = roc_auc_score(is_same_class, inner_product_scores)
        aucs.append(auc_score)
    print(model_path, np.mean(aucs))
    return np.mean(aucs)

def run_single_input_attack_task(
        model_factory: SplitModelFactory,
        model_path: str,
        get_modifier: Callable[[List[int]], Modifier], split_layer: int,
        dataset: Dataset,
        n_regenerate_samples: int = 100,
        result_output_path: str = None
):
    result_output_path = result_output_path or model_path[:-4] + "-reconstruct"

    test_sample_batch = [Convert.to_tensor(dataset[i][0]) for i in range(n_regenerate_samples)]
    test_sample_batch = torch.stack(test_sample_batch, dim=0)

    regenerated_samples = nn.Parameter(torch.normal(0, 1, test_sample_batch.shape, device=GlobalConfig.device))



    modifier = get_modifier(model_factory.get_splits()[split_layer - 1])
    split_model = model_factory.get_modified_model(split_layer, modifier)
    split_model.load_state_dict(torch.load(model_path))
    Convert.model_to_device(split_model)

    bottom_model = model_factory.get_bottom_model()
    for m in bottom_model.parameters():
        m.requires_grad = False

    optimizer = SGD([regenerated_samples], 0.01, momentum=0.9)

    for i in range(10000):
        random_indices = np.random.choice(n_regenerate_samples, int(n_regenerate_samples / 2))
        feature_mse = torch.sum(torch.square(modifier(bottom_model(test_sample_batch[random_indices])) -
                                             modifier(bottom_model(regenerated_samples[random_indices])))) \
                      / len(random_indices)
        if i % 100 == 0:
            print(f"Round {i}, loss {feature_mse.item():.4f}")
        optimizer.zero_grad()
        feature_mse.backward()
        optimizer.step()

    Path(result_output_path).mkdir(exist_ok=True)
    os.chdir(result_output_path)

    for i in range(n_regenerate_samples):
        origin_image = np.moveaxis(Convert.to_numpy(test_sample_batch[i]), 0, 2)
        plt.imshow(origin_image)
        plt.savefig(f"{i}-origin.png")

        regen_image = np.moveaxis(Convert.to_numpy(regenerated_samples[i]), 0, 2)
        plt.imshow(regen_image)
        plt.savefig(f"{i}-regen.png")


def run_single_generation_attack_cifar(
        model_factory: SplitModelFactory,
        model_path: str,
        get_modifier: Callable[[List[int]], Modifier], split_layer: int,
        train_set: Dataset,
        test_set: Dataset,
        n_regenerate_samples: int = 100,
        result_output_path: str = None
):
    modifier = get_modifier([128])
    split_model = model_factory.get_modified_model(split_layer, modifier)
    split_model.load_state_dict(torch.load(model_path))
    Convert.model_to_device(split_model)

    bottom_model = model_factory.get_bottom_model()
    for m in bottom_model.parameters():
        m.requires_grad = False

    get_generator_model = lambda: nn.Sequential(
        nn.Linear(128, 1024), nn.Sigmoid(),     # [batch, 256]
        ViewLayer([-1, 4, 16, 16]),         # [batch, 4, 16, 16]
        nn.Conv2d(4, 16, kernel_size=(3, 3), padding=(1, 1)),  # [batch, 16, 16, 16]
        nn.LeakyReLU(),
        nn.ConvTranspose2d(16, 32, kernel_size=(2, 2), stride=(2, 2)),  # [batch, 32, 32, 32]
        nn.LeakyReLU(),
        nn.Conv2d(32, 3, kernel_size=(3, 3), padding=(1, 1)),  # [batch, 3, 32, 32]
        nn.Sigmoid()
    )

    get_encoder_decoder_model = lambda: nn.Sequential(bottom_model, modifier, get_generator_model())

    gen_train = Train(
        get_encoder_decoder_model,
        DataLoader(IdentityDataset(train_set), 32),
        DataLoader(IdentityDataset(test_set), 256),
        None,
        nn.MSELoss(),
        lambda ps: SGD(ps, 0.01, momentum=0.9),
        [NegL2Loss()],
        n_rounds=1000,
        save_model=True,
        record_name="reconstruct"
    )
    result_output_path = result_output_path or model_path[:-4] + "-gen"
    Path(result_output_path).mkdir(exist_ok=True)
    os.chdir(result_output_path)

    gen_train.train()

    # Regenerate some images and see the performance
    test_sample_batch = [Convert.to_tensor(test_set[i][0]) for i in range(n_regenerate_samples)]
    test_sample_batch = torch.stack(test_sample_batch, dim=0)

    with torch.no_grad():
        regen_samples = gen_train.model(test_sample_batch)

    for i in range(n_regenerate_samples):
        origin_image = np.moveaxis(Convert.to_numpy(test_sample_batch[i]), 0, 2)
        plt.imsave(f"{i}-origin.png", origin_image)


        regen_image = np.moveaxis(Convert.to_numpy(regen_samples[i]), 0, 2)
        plt.imsave(f"{i}-regen.png", regen_image)