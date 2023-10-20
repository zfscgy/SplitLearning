from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class ParallelCNN(nn.Module):
    def __init__(self, in_channels: int, input_len: int, kernel_sizes: List[int], out_channels: int = None):
        super(ParallelCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or self.in_channels
        self.input_len = input_len
        self.kernel_sizes = kernel_sizes
        self.multi_convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels, out_channels, (filter_size,)),
            # [batch, out_channels, input_len]
            nn.MaxPool1d(input_len - filter_size + 1),
            # [batch, out_channels, 1]
            nn.ReLU()
        ) for filter_size in kernel_sizes])

    def forward(self, xs: torch.Tensor):
        xs = torch.swapaxes(xs, 1, 2)  # change to [batch, channel, len]
        conv_outs = [conv(xs) for conv in self.multi_convs]
        cated_out = torch.cat(conv_outs, dim=1)
        cated_out = torch.squeeze(cated_out)
        return cated_out  # [batch, out_channels]


def make_textcnn_modules(input_len: int, output_dim: int, vocab_size: int, embedding_dim: int=50, n_channels: int = 200,
                         kernel_sizes: List[int] = None,
                         initial_embedding: dict=None, word_map: dict=None):
    """
    :param output_dim: dimension of the output
    :param word_map: mapping from word to index
    :param embedding_dim: (
    :param filters:
    :param kernel_sizes:
    :param glove_embedding_path:
    :return:
    """

    embedding_layer = nn.Embedding(vocab_size, embedding_dim)  # '+1' is for the padded token
    for word in initial_embedding:
        if word in word_map:
            with torch.no_grad():
                embedding_layer.weight[word_map[word]] = torch.from_numpy(initial_embedding[word])
    # Initialize the word embeddings

    multi_conv = ParallelCNN(embedding_dim, input_len, kernel_sizes, n_channels)
    classifier = nn.Linear(len(kernel_sizes) * n_channels, output_dim)

    return [nn.Sequential(embedding_layer, multi_conv), classifier]
