from typing import List

import torch
from torch import nn


class MultipleEmbeddingLayer(nn.Module):
    def __init__(self, categorical_features: List[int], embedding_dim: int):
        super(MultipleEmbeddingLayer, self).__init__()
        self.categorical_features = categorical_features
        self.embedding_dim = embedding_dim
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(n_categories, embedding_dim) for n_categories in categorical_features
        ])
        self.out_dim = len(categorical_features) * embedding_dim

    def forward(self, x: torch.LongTensor):
        embeddings = [embedding_layer(x[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
        # N * [batch, embedding_dim]
        concated_embedding = torch.cat(embeddings, dim=1)
        return concated_embedding


class DeepNet(nn.Module):
    def __init__(self, categorical_features: List[int], dnn_layers: List[nn.Module], embedding_dim: int):
        super(DeepNet, self).__init__()
        self.sequential_modules = [
            MultipleEmbeddingLayer(categorical_features, embedding_dim)
        ] + dnn_layers

        self.main_module = nn.Sequential(*self.sequential_modules)

    def forward(self, x: torch.LongTensor):
        return self.main_module(x)


class WideAndDeepNet(nn.Module):
    def __init__(self, n_numeric_features: int, categorical_features: List[int], dnn_layer_sizes: List[int],
                 embedding_dim: int = 32):
        super(WideAndDeepNet, self).__init__()
        self.n_numeric_features = n_numeric_features
        self.categorical_features = categorical_features
        self.embedding_dim = embedding_dim

        dnn_indim = len(self.categorical_features) * embedding_dim
        dnn_layers = []
        for i, d in enumerate(dnn_layer_sizes):
            if i == 0:
                dnn_layers.append(nn.Sequential(nn.Linear(dnn_indim, dnn_layer_sizes[0]), nn.LeakyReLU()))
            elif i < len(dnn_layer_sizes) - 1:
                dnn_layers.append(nn.Sequential(nn.Linear(dnn_layer_sizes[i - 1], dnn_layer_sizes[i]), nn.LeakyReLU()))
            else:
                dnn_layers.append(nn.Sequential(nn.Linear(dnn_layer_sizes[i - 1], dnn_layer_sizes[i]), nn.Tanh()))

        self.deep_model = DeepNet(categorical_features, dnn_layers, embedding_dim)
        self.prediction_model = nn.Linear(dnn_layer_sizes[-1] + n_numeric_features, 1)

    def forward(self, x: List[torch.Tensor]):
        x_numerical, x_categorical = x
        deep_out = self.deep_model(x_categorical)  # [Batch, 32]
        concat_deep_wide = torch.concat([deep_out, x_numerical], dim=1)
        return torch.sigmoid(self.prediction_model(concat_deep_wide))
