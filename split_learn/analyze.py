from typing import Union, List, Dict, Callable
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad


from split_learn.modifiers import TopKModifier, RandomTopKModifier, Modifier
from split_learn.train import SplitTask

from deep_utils.convert import Convert
from deep_utils.train import Train, RoundMetricCallback, test_on_data_loader


class LossOnTrainSet(RoundMetricCallback):
    def __call__(self, train: Train):
        losses = []
        train.model.eval()

        for x, y in train.train_loader:
            losses.append(train.loss_func(train.model(Convert.to_tensor(x)), Convert.to_tensor(y)).item() * x.shape[0])
        return {
            "trainset_loss": np.sum(losses) / len(train.train_loader.dataset)
        }


class MetricOnTrainSet(RoundMetricCallback):
    def __call__(self, train: Train):
        train.model.eval()

        trainset_metrics = test_on_data_loader(train.model, train.train_loader, train.metrics)
        renamed_metrics = dict()

        for key in trainset_metrics:
            renamed_metrics["trainset_" + key] = trainset_metrics[key]

        return renamed_metrics



class OrderChange(RoundMetricCallback):
    """
    To record the distribution of non-zero entries in mask,
    i.e., find out which neurons are selected most
    """
    def __init__(self, sparsity: float, split_layer: int = -2, n_batches: int = 10):
        self.sparsity = sparsity
        self.split_layer = split_layer
        self.n_batches = n_batches

        self.bottom_model = None
        self.data_loader = None

        self.previous_order = None
        self.current_order = None

    def __call__(self, train: Train):
        if self.bottom_model is None:
            self.bottom_model = nn.Sequential(*list(train.model.children())[:self.split_layer])
        self.bottom_model.eval()

        self.data_loader = train.test_loader

        b = 0
        hs_orders = []
        for xs, _ in self.data_loader:
            if b == self.n_batches:
                break
            b += 1
            with torch.no_grad():
                hs = self.bottom_model(Convert.to_tensor(xs))
            hs = hs.view(hs.shape[0], -1)
            hs_argindex = torch.argsort(torch.abs(hs))
            hs_order = torch.argsort(hs_argindex)
            # Get the inverse permutation of argindex,
            # since we have to get the order of elements, instead of the index of the sorted elements

            hs_orders.append(hs_order)

        for _ in self.data_loader:  # Enumerate to the end
            pass

        self.previous_order = self.current_order
        self.current_order = torch.cat(hs_orders, dim=0)

        threshold = self.current_order.shape[1] - round(self.sparsity * self.current_order.shape[1])

        if self.previous_order is None:
            return {
                "absOrderChange": np.nan,
                f"absOrderChange@{threshold}": np.nan,
                f"orderChange@{threshold}": np.nan
            }
        else:

            threshold_mask = (self.previous_order >= threshold).float()
            threshold_sum = torch.sum(threshold_mask)
            order_change = self.current_order.float() - self.previous_order.float()

            return {
                "absOrderChange": torch.mean(torch.abs(order_change)).item(),
                f"absOrderChange@{threshold}": (torch.sum(torch.abs(order_change) * threshold_mask) / threshold_sum).item(),
                f"orderChange@{threshold}": (torch.sum(order_change * threshold_mask) / threshold_sum).item()
            }


class TopkReplacements(RoundMetricCallback):
    def __init__(self, sparsity: float, split_layer: int = -2, n_batches: int = 10):

        self.sparsity = sparsity
        self.split_layer = split_layer

        self.n_batches = n_batches

        self.bottom_model = None
        self.data_loader = None

        self.previous_mask = None
        self.current_mask = None
        self.topk_modifier = None

    def __call__(self, train: Train):
        if self.bottom_model is None:
            self.bottom_model = nn.Sequential(*list(train.model.children())[:self.split_layer])
        self.bottom_model.eval()

        self.data_loader = train.test_loader

        mask_list = []
        b = 0
        for xs, _ in self.data_loader:
            if b == self.n_batches:
                break
            b += 1
            with torch.no_grad():
                hs = self.bottom_model(Convert.to_tensor(xs))
                if self.topk_modifier is None:
                    self.topk_modifier = TopKModifier(self.sparsity, hs.shape[1:])
                self.topk_modifier(hs)
            mask_list.append(self.topk_modifier.mask.view(hs.shape[0], -1))

        for _ in self.data_loader:  # Enumerate to the end
            pass

        self.previous_mask = self.current_mask
        self.current_mask = torch.cat(mask_list, dim=0)
        if self.previous_mask is None:
            return {
                f"topkReplacement-top{self.sparsity}": np.nan
            }
        else:
            replacement = (torch.mean(torch.abs(self.current_mask.float() - self.previous_mask.float()) *
                                      self.previous_mask)).item() * self.previous_mask.shape[1]
            return {
                f"topkReplacement-top{self.sparsity}": replacement
            }


class NeuronChange(RoundMetricCallback):
    def __init__(self, sparsity: float, split_layer: int = -2, n_batches: int = 10):
        self.sparsity = sparsity
        self.split_layer = split_layer
        self.n_batches = n_batches

        self.bottom_model = None
        self.data_loader = None
        self.topk_modifier = None

        self.previous_neurons = None
        self.change_large = None
        self.change_small = None

    def __call__(self, train: Train):
        if self.bottom_model is None:
            self.bottom_model = nn.Sequential(*list(train.model.children())[:self.split_layer])
        self.bottom_model.eval()

        self.data_loader = train.test_loader

        neuron_list_large = []
        neuron_list_small = []
        b = 0
        for xs, _ in self.data_loader:
            if b == self.n_batches:
                break
            b += 1
            with torch.no_grad():
                hs = self.bottom_model(Convert.to_tensor(xs))
                if self.topk_modifier is None:
                    self.topk_modifier = TopKModifier(self.sparsity, hs.shape[1:])
                self.topk_modifier(hs)
            batch_size = hs.shape[0]
            mask = self.topk_modifier.mask.view(batch_size, -1).float()
            output = hs.view(batch_size, -1)
            neuron_list_large.append(output * mask)
            neuron_list_small.append(output * (1 - mask))

        for _ in self.data_loader:  # Enumerate to the end
            pass

        neuron_list_large = torch.cat(neuron_list_large, dim=0)
        neuron_list_small = torch.cat(neuron_list_small, dim=0)

        if self.previous_neurons is None:
            self.previous_neurons = (neuron_list_large, neuron_list_small)
            change_large = np.nan
            change_small = np.nan
        else:
            change_large = torch.mean(torch.square(neuron_list_large - self.previous_neurons[0])).item()
            change_small = torch.mean(torch.square(neuron_list_small - self.previous_neurons[1])).item()

        self.previous_neurons = (neuron_list_large, neuron_list_small)

        return {
            "largeNeuronChange": change_large,
            "smallNeuronChange": change_small
        }


class NeuronInteraction(RoundMetricCallback):
    def __init__(self, split_layer: int = -2, n_samples: int = 100):
        self.split_layer = split_layer
        self.n_samples = n_samples

        self.bottom_model = None
        self.dataset = None

        self.stats = []
        self.other_stats = []

    def __call__(self, train: Train):
        self.dataset = train.test_loader.dataset
        if self.bottom_model is None:
            self.bottom_model = nn.Sequential(*list(train.model.children())[:self.split_layer])
        self.bottom_model.eval()

        in_sample_self = []
        in_sample_cross = []
        cross_sample_self = []
        cross_sample_cross = []

        for i in range(self.n_samples):
            idx0, idx1 = np.random.choice(len(self.dataset), 2, replace=False)
            sample_0 = Convert.to_tensor(self.dataset[idx0][0].unsqueeze(0))
            sample_1 = Convert.to_tensor(self.dataset[idx1][0].unsqueeze(0))

            output_0 = self.bottom_model(sample_0).flatten()
            output_1 = self.bottom_model(sample_1).flatten()

            idx0, idx1 = np.random.choice(output_0.shape[0], 2, replace=False)

            neuron00 = output_0[idx0]  # sample 0, neuron 0
            neuron01 = output_0[idx1]  # sample 0, neuron 1
            neuron10 = output_1[idx0]
            neuron11 = output_1[idx1]

            grad_00 = grad(neuron00, self.bottom_model.parameters(), retain_graph=True)
            grad_01 = grad(neuron01, self.bottom_model.parameters(), retain_graph=True)
            grad_10 = grad(neuron10, self.bottom_model.parameters(), retain_graph=True)
            grad_11 = grad(neuron11, self.bottom_model.parameters())

            def inner_product(x, y):
                return torch.sum(torch.stack([torch.sum(p0 * p1) for p0, p1 in zip(x, y)]))

            in_sample_self.append(inner_product(grad_00, grad_00))
            in_sample_cross.append(inner_product(grad_00, grad_01))
            cross_sample_self.append(inner_product(grad_00, grad_10))
            cross_sample_cross.append(inner_product(grad_00, grad_11))

        in_sample_self = torch.mean(torch.stack(in_sample_self), dim=0)
        in_sample_cross = torch.mean(torch.stack(in_sample_cross), dim=0)
        cross_sample_self = torch.mean(torch.stack(cross_sample_self), dim=0)
        cross_sample_cross = torch.mean(torch.stack(cross_sample_cross), dim=0)
        interactions_dict = dict()

        interactions_dict[f'iis'] = in_sample_self.item()
        interactions_dict[f'iic'] = in_sample_cross.item()
        interactions_dict[f'ics'] = cross_sample_self.item()
        interactions_dict[f'icc'] = cross_sample_cross.item()

        return interactions_dict


class NeuronCoAdaption(RoundMetricCallback):
    def __init__(self, split_layer: int = -2, n_batches: int = 100):
        self.split_layer = split_layer
        self.n_batches = n_batches

        self.bottom_model = None
        self.dataset = None


    def __call__(self, train: Train):
        self.dataset = train.test_loader.dataset
        if self.bottom_model is None:
            self.bottom_model = nn.Sequential(*list(train.model.children())[:self.split_layer])
        self.bottom_model.eval()
        self.data_loader = train.test_loader

        all_hs = []
        b = 0
        for xs, _ in self.data_loader:
            if b == self.n_batches:
                break
            b += 1
            with torch.no_grad():
                hs = self.bottom_model(Convert.to_tensor(xs))
                hs = hs.view(hs.shape[0], -1)
                all_hs.append(hs)

        for _ in self.data_loader:  # Enumerate to the end
            pass

        all_hs = torch.cat(all_hs, dim=0)  # [N, size]
        correlation = torch.corrcoef(all_hs.transpose(0, 1))  # [size, size]
        hs_dim = correlation.shape[0]
        mask = 1 - Convert.to_tensor(torch.eye(hs_dim))

        abs_correlation = torch.abs(correlation * mask)
        max_co_adaption = torch.max(abs_correlation)
        co_adaption = torch.sum(abs_correlation) / (hs_dim ** 2 - hs_dim)

        return {
            "neuronCoAdaption": co_adaption.item(),
            "neuronCoAdaptionMax": max_co_adaption.item()
        }


class LastWeightNorm(RoundMetricCallback):
    def __init__(self, split_layer: int = -2):
        self.split_layer = split_layer

        self.last_weight = None
        self.dataset = None

    def __call__(self, train: Train):
        if self.last_weight is None:
            self.last_weight = list(train.model.parameters())[-2]

        l2norms = torch.norm(self.last_weight, 2, dim=1)
        return {
            "lastWeightL2norm": torch.mean(l2norms).item()
        }


class SingleSampleLossChange(RoundMetricCallback):
    """
    Single samples' loss change on
    """
    def __init__(self, modifier: TopKModifier, split_layer: int = -2, n_batches: int = 10, loss_func=None):
        self.topk_modifier = modifier

        self.split_layer = split_layer
        self.n_batches = n_batches

        self.loss_func = loss_func or nn.CrossEntropyLoss(reduction='none')

        self.bottom_model = None
        self.dataset = None

        self.previous_indices = None
        self.previous_losses = None

    def __call__(self, train: Train):
        if self.bottom_model is None:
            self.bottom_model = nn.Sequential(*list(train.model.children())[:self.split_layer])
        self.bottom_model.eval()
        self.data_loader = train.train_loader

        current_indices = []
        current_losses = []
        self.bottom_model.eval()

        b = 0
        for xs, ys in self.data_loader:
            if b < self.n_batches:
                b += 1
            else:
                break
            with torch.no_grad():
                current_batch_loss = self.loss_func(train.model(Convert.to_tensor(xs)), Convert.to_tensor(ys))
                current_losses.append(current_batch_loss)

                self.topk_modifier(self.bottom_model(Convert.to_tensor(xs)))
                current_indices.append(self.topk_modifier.mask)
        for _ in self.data_loader:
            pass

        current_losses = torch.cat(current_losses, dim=0)
        current_indices = torch.cat(current_indices, dim=0)
        current_indices = current_indices.view(current_indices.shape[0], -1)

        metrics = dict()

        if self.previous_losses is not None:
            for i in range(self.topk_modifier.non_zero_elements + 1):
                mask_reserved_i = (torch.sum(current_indices * self.previous_indices, dim=1) == i).float()
                avg_loss_change = torch.sum((current_losses - self.previous_losses) * mask_reserved_i) / torch.sum(mask_reserved_i)

                metrics[f"loss_change@reserved_{i}"] = avg_loss_change.item()
                metrics[f"ratio_reservered_{i}"] = torch.sum(mask_reserved_i).item() / mask_reserved_i.shape[0]
        else:
            for i in range(self.topk_modifier.non_zero_elements + 1):
                metrics[f"loss_change@reserved_{i}"] = np.nan
                metrics[f"ratio_reservered_{i}"] = np.nan

        self.previous_losses = current_losses
        self.previous_indices = current_indices

        return metrics


"""
export CUDA_VISIBLE_DEVICES=0
nohup python -u sparse_analyze.py -i 0 > z0.log &
nohup python -u sparse_analyze.py -i 1 > z1.log &
export CUDA_VISIBLE_DEVICES=1
nohup python -u sparse_analyze.py -i 2 > z2.log &
nohup python -u sparse_analyze.py -i 3 > z3.log &
"""