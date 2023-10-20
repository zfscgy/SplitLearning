from typing import Tuple

from sklearn.metrics import roc_auc_score

import torch


from deep_utils.convert import Convert


class Metric:
    def __init__(self, name: str, use_batch_mean: bool = True):
        self.name = name
        self.use_batch_mean = use_batch_mean

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        raise NotImplemented()


class BinAcc(Metric):
    def __init__(self):
        super(BinAcc, self).__init__("acc", True)

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        return torch.mean((torch.round(pred_ys) == ys).float())


class ClassAcc(Metric):
    def __init__(self):
        super(ClassAcc, self).__init__("acc", True)

    def compute(self, pred_ys: torch.Tensor, ys: torch.LongTensor):
        return torch.mean((torch.argmax(pred_ys, dim=-1) == ys).float())


class Auc(Metric):
    def __init__(self):
        super(Auc, self).__init__("auc", False)

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        return roc_auc_score(Convert.to_numpy(ys[:, 0]), Convert.to_numpy(pred_ys[:, 0]))


class HitRatio(Metric):
    def __init__(self, n: int):
        super(HitRatio, self).__init__(f"hr@{n}", True)
        self.n = n

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        _, indices = torch.sort(pred_ys, dim=-1)
        top_n_indices = indices[:, -self.n:]  # [batch, n]
        return torch.mean(torch.sum((ys.view(-1, 1) == top_n_indices).float(), dim=-1))


class NegL2Loss(Metric):
    def __init__(self):
        super(NegL2Loss, self).__init__("negL2", True)

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        return -torch.sum(torch.square(pred_ys - ys)) / ys.shape[0]


class Sparsity(Metric):
    def __init__(self, abs_threshold: float = 1e-3):
        super(Sparsity, self).__init__("sparsity", True)
        self.abs_thrshold = abs_threshold

    def compute(self, pred_ys: torch.Tensor, _: torch.Tensor):
        return torch.sum(torch.abs(pred_ys) > self.abs_thrshold) / pred_ys.flatten().shape[0]


class MultiOutputMetric(Metric):
    def __init__(self, base_metric: Metric, index_0: int, index_1: int):
        super(MultiOutputMetric, self).__init__(str(index_0) + '-' + str(index_1) + base_metric.name, base_metric.use_batch_mean)
        self.base_metric = base_metric
        self.index_0 = index_0
        self.index_1 = index_1

    def compute(self, pred_ys: Tuple[torch.Tensor, ...], ys: Tuple[torch.Tensor, ...]):
        if not isinstance(ys, Tuple):
            ys = (ys,)
        return self.base_metric.compute(pred_ys[self.index_0], ys[self.index_1])


if __name__ == '__main__':
    print(Sparsity().compute(torch.normal(0, 1, [100]) / 3000, None))

