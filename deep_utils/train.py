import os
from datetime import datetime
from typing import List, Iterator, Callable, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from deep_utils.convert import Convert
from deep_utils.metrics import Metric


def train_n_batches(model: nn.Module, optimizer: Optimizer, loss_func: Callable,
                    data_loader: DataLoader, data_iterator: Iterator, n_batches: int):

    i = 0
    losses = []
    while i < n_batches:
        try:
            xs, ys = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            xs, ys = next(data_iterator)
        pred_ys = model(Convert.to_tensor(xs))
        loss = loss_func(pred_ys, Convert.to_tensor(ys))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        i += 1

    return losses, data_iterator


def train_n_epochs(model: nn.Module, optimizer: Optimizer, loss_func: Callable,
                   data_loader: DataLoader, n_epochs: int):
    losses = []
    for i in range(n_epochs):
        for xs, ys in data_loader:
            pred_ys = model(Convert.to_tensor(xs))
            loss = loss_func(pred_ys, Convert.to_tensor(ys))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return losses


def test_on_data_loader(model: Callable, data_loader: DataLoader, metrics: List[Metric]) -> Dict[str, float]:
    ys = []
    pred_ys = []
    metric_vals = dict()

    global_metric = False
    for metric in metrics:
        metric_vals[metric.name] = []
        if not metric.use_batch_mean:
            global_metric = True

    n_total_samples = 0
    for batch_xs, batch_ys in data_loader:
        batch_xs = Convert.to_tensor(batch_xs)
        batch_ys = Convert.to_tensor(batch_ys)

        with torch.no_grad():
            batch_pred_ys = model(batch_xs)
            for i, metric in enumerate(metrics):
                if metric.use_batch_mean:
                    metric_vals[metric.name].append(metric.compute(batch_pred_ys, batch_ys) * len(batch_xs))
            n_total_samples += len(batch_xs)
        if global_metric:
            ys.append(batch_ys)
            pred_ys.append(batch_pred_ys)

    if global_metric:
        if not isinstance(ys[0], torch.Tensor):  # In case of multiple outputs
            ys = [torch.cat([y[i] for y in ys]) for i in range(len(ys[0]))]
        else:
            ys = torch.cat(ys, dim=0)

        if not isinstance(pred_ys[0], torch.Tensor):
            pred_ys = [torch.cat([y[i] for y in pred_ys]) for i in range(len(pred_ys[0]))]
        else:
            pred_ys = torch.cat(pred_ys, dim=0)

    for i, metric in enumerate(metrics):
        if not metric.use_batch_mean:
            # Compute at the same time
            metric_vals[metric.name] = metric.compute(pred_ys, ys)
        else:
            metric_vals[metric.name] = torch.sum(torch.stack(metric_vals[metric.name], dim=0)).item() / n_total_samples

    return metric_vals


class Train:
    def __init__(self, get_model: Callable[[], nn.Module],
                 train_loader: DataLoader, validation_loader: DataLoader, test_Loader: DataLoader,
                 loss_func: Callable, get_optimizer: Callable[[Any], Optimizer],
                 metrics: List[Metric], callback_metrics: List[Callable] = None,
                 n_rounds: int = 100, batches_per_round: int = None, round_callback: Callable = None,
                 early_stop: int = 0, until_loss: float = -1e8, save_model: bool = False,
                 record_name: str = None
                 ):
        self.get_model = get_model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_Loader

        self.loss_func = loss_func
        self.get_optimizer = get_optimizer

        self.metrics = metrics
        self.callback_metrics = callback_metrics

        self.current_round = 0


        self.batches_per_round = None

        self.n_rounds = n_rounds
        self.batches_per_round = batches_per_round
        self.round_callback = round_callback

        self.early_stop = early_stop
        self.until_loss = until_loss
        self.save_model = save_model

        self.record_name = record_name or "record"

        self.model = None
        self.optimizer = None


    def train(self):
        self.model = Convert.model_to_device(self.get_model())
        self.optimizer = self.get_optimizer(self.model.parameters())


        train_record = None
        train_iterator = iter(self.train_loader)
        current_train_loss = 1e8

        best_main = -1e8
        for i in range(self.n_rounds):
            self.model.eval()
            metric_dict = dict()
            if self.batches_per_round is not None:
                metric_dict['iterations'] = i * self.batches_per_round
            else:
                metric_dict['epochs'] = i

            metric_dict.update(test_on_data_loader(self.model, self.validation_loader, self.metrics))

            if self.callback_metrics is not None:
                for m in self.callback_metrics:
                    metric_dict.update(m(self))

            metric_dict["TrainLoss"] = current_train_loss

            print(f"Before round {i}, metric values: {metric_dict}")
            if train_record is None:
                train_record = pd.DataFrame(columns=metric_dict.keys())
            train_record.loc[len(train_record)] = metric_dict
            if self.record_name is not None:
                train_record.to_csv(self.record_name + ".csv")

            main_metric = self.metrics[0].name

            if self.early_stop != 0:
                if i >= 2 * self.early_stop:
                    previous_mean = train_record.loc[i - 2 * self.early_stop: i - self.early_stop, main_metric].mean()
                    new_mean = train_record.loc[i - self.early_stop: i, main_metric].mean()
                    if previous_mean >= new_mean:
                        print(f"Early stop triggered: {new_mean} <= {previous_mean}")
                        break

            if metric_dict[main_metric] > best_main and i != 0:
                print("Best performance achieved, save model...")
                best_main = metric_dict[main_metric]
                torch.save(self.model.state_dict(), self.record_name + ".pth")

            self.model.train()
            if self.batches_per_round is not None:
                losses, train_iterator = \
                    train_n_batches(self.model, self.optimizer, self.loss_func,
                                    self.train_loader, train_iterator, self.batches_per_round)
            else:
                losses = train_n_epochs(self.model, self.optimizer, self.loss_func, self.train_loader, 1)

            if current_train_loss < self.until_loss:
                break
            current_train_loss = np.mean(losses)

            if self.round_callback is not None:
                self.round_callback(self)
            self.current_round += 1

        # Load the best model
        self.model.load_state_dict(torch.load(self.record_name + ".pth"))

        if self.test_loader is not None:
            self.model.eval()
            metric_dict = dict()
            if self.batches_per_round is not None:
                metric_dict['iterations'] = -1
            else:
                metric_dict['epochs'] = -1

            metric_dict.update(test_on_data_loader(self.model, self.test_loader, self.metrics))
            train_record.loc[len(train_record)] = metric_dict
            if self.record_name is not None:
                train_record.to_csv(self.record_name + ".csv")
            print(f"Metric values on test set: {metric_dict}")

        if not self.save_model:
            os.remove(self.record_name + ".pth")
        return train_record


class RoundMetricCallback:
    def __call__(self, train: Train):
        raise NotImplementedError()


class RoundMetricsRecorder(RoundMetricCallback):
    def __init__(self, round_metric_callbacks: List[RoundMetricCallback]):
        self.callbacks = round_metric_callbacks
        self.record = None

    def __call__(self, train: Train):
        record_name = train.record_name + "-roundCallbacks"
        round_metrics = dict()
        for callback in self.callbacks:
            round_metrics.update(callback.__call__(train))
        if self.record is None:
            self.record = pd.DataFrame(columns=round_metrics.keys())
        self.record.loc[len(self.record)] = round_metrics
        self.record.to_csv(record_name + ".csv")


def gen_param_str(prefix: str, params: dict):
    return prefix + "-" + "_".join([key + "_" + str(params[key]) for key in params])
