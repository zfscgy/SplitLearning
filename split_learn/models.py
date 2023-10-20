import time
from typing import Tuple, List, Callable
from pathlib import Path

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from split_learn.modifiers import Modifier
from split_learn.nn.resnet import make_resnet20_modules
from split_learn.nn.efficient_net import make_efficient_net_b0_64x64_modules
from split_learn.nn.gru4rec import make_gru4rec_modules
from split_learn.nn.wide_and_deep import WideAndDeepNet
from split_learn.nn.textcnn import make_textcnn_modules


class SplitModelFactory(nn.Module):
    def __init__(self):
        super(SplitModelFactory, self).__init__()

    def get_splits(self) -> List[List[int]]:
        """
        :return: output shapes of different splits. Notice: split position starts from 1
        """
        raise NotImplementedError()

    def get_modified_model(self, split_position: int, modifier: Modifier, seed: int = int(time.time())) -> nn.Module:
        raise NotImplementedError()

    def get_bottom_model(self) -> nn.Module:
        raise NotImplementedError()


class SequentialSplitFactory(SplitModelFactory):
    def __init__(self, get_modules: Callable[[], List[nn.Module]], output_shapes: List[List[int]]):
        super(SequentialSplitFactory, self).__init__()
        self.get_modules = get_modules
        self.output_shapes = output_shapes

        self.module_list = None
        self.split_position = -1

    def get_splits(self) -> List[List[int]]:
        return self.output_shapes

    def get_modified_model(self, split_position: int, modifier: Modifier, seed: int = None) -> nn.Module:
        self.split_position = split_position
        if seed is not None:
            torch.manual_seed(seed)
        self.module_list = self.get_modules()
        return nn.Sequential(
            *self.module_list[:split_position],
            modifier,
            *self.module_list[split_position:]
        )

    def get_bottom_model(self):
        return nn.Sequential(*self.module_list[:self.split_position])


class MnistDNNFactory(SequentialSplitFactory):
    def __init__(self):
        super(MnistDNNFactory, self).__init__(lambda: [
            nn.Sequential(nn.Linear(784, 128), nn.LeakyReLU()),
            nn.Sequential(nn.Linear(128, 32), nn.Tanh()),
            nn.Linear(32, 10)
        ], [[128], [32]])


class Resnet20Factory(SequentialSplitFactory):
    def __init__(self, outdim: int = 10):
        super(Resnet20Factory, self).__init__(
            lambda: make_resnet20_modules(outdim),
            [
                [16, 32, 32],
                [32, 32, 32],
                [64, 16, 16],
                [128, 8, 8],
                [128],
            ]
        )


class EfficientNetFactory(SequentialSplitFactory):
    def __init__(self, outdim: int=200, pretrained: bool=False):
        super(EfficientNetFactory, self).__init__(
            lambda: make_efficient_net_b0_64x64_modules(outdim, pretrained=pretrained),
            [
                [32, 32, 32],   # 1
                [16, 32, 32],   # 2
                [24, 16, 16],   # 3
                [40, 8, 8],     # 4
                [80, 4, 4],     # 5
                [112, 4, 4],    # 6
                [192, 2, 2],    # 7
                [320, 2, 2],    # 8
                [1280, 2, 2],   # 9
                [1280]          # 10
            ]
        )


class TextCNNFactory(SequentialSplitFactory):
    def __init__(self, vocab_size: int, output_dim: int, input_len: int = 100, embedding_dim: int = 50, n_channels: int = 200,
                        kernel_sizes: List[int] = None, initial_embedding: dict = None, word_map: dict = None):
        kernel_sizes = kernel_sizes or [3, 4, 5]
        super(TextCNNFactory, self).__init__(
            lambda: make_textcnn_modules(input_len, output_dim, vocab_size, embedding_dim, n_channels, kernel_sizes,
                                         initial_embedding, word_map),
            [[n_channels * len(kernel_sizes)]]
        )


class GRU4RecFactory(SequentialSplitFactory):
    def __init__(self, n_items: int, embedding_dim: int, gru_hidden_size: int, additional_linear: int = None):
        split_layer_size = [gru_hidden_size]
        if additional_linear:
            split_layer_size = [additional_linear]
        super(GRU4RecFactory, self).__init__(lambda: make_gru4rec_modules(
            n_items, embedding_dim, gru_hidden_size, n_grus=1, additional_linear=additional_linear
        ), [split_layer_size])


class WideAndDeepFactory(SplitModelFactory):
    def __init__(self, n_numeric_features: int, categorical_features: List[int], dnn_layer_sizes: List[int],
                 embedding_dim: int = 16):
        super(WideAndDeepFactory, self).__init__()
        self.get_wide_and_deep = lambda: WideAndDeepNet(n_numeric_features, categorical_features, dnn_layer_sizes, embedding_dim)
        self.deep_factory = SequentialSplitFactory(
            lambda: self.wide_and_deep.deep_model.sequential_modules,
            [39 * embedding_dim] + [[d] for d in dnn_layer_sizes],
        )
        self.wide_and_deep = None

    def get_splits(self) -> List[List[int]]:
        return self.deep_factory.get_splits()

    def get_modified_model(self, split_position: int, modifier: Modifier, seed: int = int(time.time())) -> nn.Module:
        torch.random.manual_seed(seed)
        self.wide_and_deep = self.get_wide_and_deep()
        self.wide_and_deep.deep_model.mlp = self.deep_factory.get_modified_model(split_position, modifier, seed)
        return self.wide_and_deep

    def get_bottom_model(self):
        return self.deep_factory.get_bottom_model()


class LambdaModule(nn.Module):
    def __init__(self, original_module: nn.Module, forward_transform: Callable):
        super(LambdaModule, self).__init__()
        self.original_module = original_module
        self.forward_transform = forward_transform
    
    def forward(self, x):
        y = self.original_module(x)
        return self.forward_transform(x, y)
        

class InputPreservingModule(LambdaModule):
    def __init__(self, original_module):
        super(InputPreservingModule, self).__init__(original_module, (lambda x, y: (y, x)))



class MultiOutputModel(nn.Module):
    def __init__(self, modules: List[nn.Module], split_position: int):
        super(MultiOutputModel, self).__init__()
        self.modules = modules
        self.bottom_model = nn.Sequential(*modules[:split_position])
        self.top_model = InputPreservingModule(nn.Sequential(*modules[split_position:]))
        self.split_position = split_position

    def forward(self, x):
        h = self.bottom_model(x)
        y = self.top_model(h)
        return y


class MultiOutputModelFactory(SplitModelFactory):
    def __init__(self, get_modules: Callable[[], List[nn.Module]], output_shapes: List[List[int]]):
        super(MultiOutputModelFactory, self).__init__()
        self.get_modules = get_modules
        self.output_shapes = output_shapes

        self.module_list = None
        self.split_position = -1

    def get_splits(self) -> List[List[int]]:
        return self.output_shapes

    def get_modified_model(self, split_position: int, modifier: Modifier, seed: int = None) -> nn.Module:
        self.split_position = split_position
        torch.manual_seed(seed)
        self.module_list = self.get_modules()
        self.multi_output_model = MultiOutputModel(self.module_list, self.split_position)
        return nn.Sequential(
            self.multi_output_model.bottom_model,
            modifier,
            self.multi_output_model.top_model
        )

    def get_bottom_model(self):
        return self.multi_output_model.bottom_model


class MultiOutputResnet20Factory(MultiOutputModelFactory):
    def __init__(self, outdim: int = 10):
        super(MultiOutputResnet20Factory, self).__init__(
            partial(make_resnet20_modules, outdim=outdim),
            [
                [16, 32, 32],
                [32, 32, 32],
                [64, 16, 16],
                [128, 8, 8],
                [128],
            ])


class MultiOutputGRU4RecFactory(MultiOutputModelFactory):
    def __init__(self, n_items: int, embedding_dim: int, gru_hidden_size: int):
        super(MultiOutputGRU4RecFactory, self).__init__(
            partial(make_gru4rec_modules,
                    n_items=n_items, embedding_dim=embedding_dim, gru_hidden_size=gru_hidden_size),
            [
                [gru_hidden_size]
            ])


class MultiOutputTextCNNFactory(MultiOutputModelFactory):
    def __init__(self, vocab_size: int, output_dim: int, input_len: int = 100, 
                 embedding_dim: int = 50, n_channels: int = 200,
                 kernel_sizes: List[int] = None, initial_embedding: dict = None, word_map: dict = None):
        kernel_sizes = kernel_sizes or [3, 4, 5]
        super(MultiOutputTextCNNFactory, self).__init__(
            partial(make_textcnn_modules, vocab_size=vocab_size, output_dim=output_dim, input_len=input_len,
                    embedding_dim=embedding_dim, n_channels=n_channels, initial_embedding=initial_embedding,
                    word_map=word_map, kernel_sizes=kernel_sizes),
            [[n_channels * len(kernel_sizes)]])


class MultiOutputEfficientNetFactory(MultiOutputModelFactory):
    def __init__(self, outdim: int=200, pretrained: bool=False):
        super(MultiOutputEfficientNetFactory, self).__init__(
            partial(make_efficient_net_b0_64x64_modules, num_classes=outdim, pretrained=pretrained),
            [
                [32, 32, 32],  # 1
                [16, 32, 32],  # 2
                [24, 16, 16],  # 3
                [40, 8, 8],  # 4
                [80, 4, 4],  # 5
                [112, 4, 4],  # 6
                [192, 2, 2],  # 7
                [320, 2, 2],  # 8
                [1280, 2, 2],  # 9
                [1280]  # 10
            ]
        )

