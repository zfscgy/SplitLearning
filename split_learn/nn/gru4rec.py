import time
from typing import List


import torch
from torch import nn
import torch.nn.functional as F


class WrappedGRU(nn.Module):
    def __init__(self, gru: nn.GRU):
        super(WrappedGRU, self).__init__()
        self.gru = gru

    def forward(self, x: torch.Tensor):
        all_hiddens, last_hidden = self.gru(x)
        return last_hidden[0]


def make_gru4rec_modules(n_items: int, embedding_dim: int, gru_hidden_size: int,
                         n_grus: int = 1, additional_linear: int = None):
    input_embedding_layer = nn.Sequential(nn.Embedding(n_items, embedding_dim), nn.Dropout(p=0.25))
    gru_layers = WrappedGRU(nn.GRU(embedding_dim, gru_hidden_size, n_grus, batch_first=True))
    if additional_linear is None:
        output_layer = nn.Sequential(nn.Linear(gru_hidden_size, n_items), nn.ELU())
        return [nn.Sequential(input_embedding_layer, gru_layers), output_layer]
    else:
        output_layer = nn.Sequential(nn.Linear(additional_linear, n_items), nn.ELU())
        return [nn.Sequential(input_embedding_layer, gru_layers,
                              nn.Linear(gru_hidden_size, additional_linear), nn.Tanh()),
                output_layer]


def bpr_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    :param y_pred: [batch, n_items]
    :param y_true: [batch]
    :return:
    """
    cross_logits = y_pred[:, y_true]  # Using other samples' predictions as negative predictions.
    # [batch, batch]

    loss = - torch.sum(F.logsigmoid(cross_logits.diag().view(-1, 1) - cross_logits)) / y_true.shape[0]
    return loss
