from typing import List

from deep_utils.convert import Convert

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Modifier(nn.Module):
    def __init__(self, input_shape: List[int] = None, name: str = "identity"):
        super(Modifier, self).__init__()
        self.input_shape = input_shape
        self.compression_rate = 1
        self.name = name
        self.non_zero_elements = 0
        self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TruncationModifier(Modifier):
    def __init__(self, input_shape: List[int] = None, threshold: float=1e-3):
        super(TruncationModifier, self).__init__(input_shape, name="truncation")
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x
        else:
            x[torch.abs(x) < self.threshold] = 0
        return x

class FixedSparseModifier(Modifier):
    def __init__(self, rate: float, input_shape: List[int]):
        """
        :param rate:
        :param input_shape:
        """
        super(FixedSparseModifier, self).__init__(input_shape, name="fixed")
        self.rate = rate
        self.non_zero_elements = round(rate * np.prod(input_shape)) or 1
        self.mask = nn.Parameter(torch.zeros([np.prod(input_shape)]), requires_grad=False)
        mask_flat = self.mask.view(-1)
        non_zero_indices = torch.randperm(mask_flat.shape[0])[:self.non_zero_elements]
        mask_flat[non_zero_indices] = 1

        self.compression_rate = self.non_zero_elements / np.prod(input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        return (x_flat * self.mask).view(x.shape)


class RandomDropoutModifier(Modifier):
    """
    Note that the dropout does not ensure the non-zero elements' number is fixed.
    It is approximately about rate * N.
    """
    def __init__(self, rate: float, input_shape: List[int] = None):
        super(RandomDropoutModifier, self).__init__(input_shape, name="random")
        self.rate = rate
        self.non_zero_elements = round(rate * np.prod(input_shape)) or 1
        self.compression_rate = self.non_zero_elements / np.prod(input_shape) * \
                                (1 + np.ceil(np.log2(np.prod(input_shape))) / 32)
        self.raw_mask = torch.zeros(np.prod(input_shape))
        self.raw_mask[:self.non_zero_elements] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        hidden_size = self.raw_mask.shape[0]

        self.mask = torch.stack([self.raw_mask[torch.randperm(hidden_size)] for _ in range(batch_size)], 0)
        self.mask = self.mask.bool()
        modified_x = torch.zeros_like(x_flat, requires_grad=False)
        modified_x[self.mask] = x_flat[self.mask]
        return modified_x.view(x.shape)


class TopKModifier(Modifier):
    def __init__(self, rate: float, input_shape: List[int], batch_dims: int = 1):
        super(TopKModifier, self).__init__(input_shape, name="topk")
        self.rate = rate
        self.batch_dims = batch_dims
        self.non_zero_elements = round(rate * np.prod(input_shape)) or 1
        self.compression_rate = self.non_zero_elements / np.prod(input_shape) * \
                                (1 + np.ceil(np.log2(np.prod(input_shape))) / 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size_flat = np.prod(x.shape[:self.batch_dims])
        x_flat = x.view(batch_size_flat, -1)
        k = self.non_zero_elements
        topk_xs, _ = torch.topk(torch.abs(x_flat), k, sorted=True)
        topk_x = topk_xs[:, -1:]  # [..., 1]

        big_mask = torch.abs(x_flat) >= topk_x
        self.mask = big_mask.view(*x.shape)
        modified_x = torch.zeros_like(x, requires_grad=False)
        modified_x[self.mask] = x[self.mask]
        return modified_x


class UniformRandomTopKModifier(TopKModifier):
    def __init__(self, rate: float, input_shape: List[int], exponent: float = 3, batch_dims: int = 1):
        super(UniformRandomTopKModifier, self).__init__(rate, input_shape, batch_dims)
        self.name = f"uRandTopk{exponent}"
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size_flat = np.prod(x.shape[:self.batch_dims])
            x_flat = x.view(batch_size_flat, -1).abs()
            r = torch.rand_like(x_flat) * \
                (((x_flat / (torch.max(x_flat, dim=1, keepdim=True)[0] + 0.001))) ** self.exponent)

            k = self.non_zero_elements
            topk_xs, topk_indices = torch.topk(r, k, sorted=True)

            big_mask = torch.zeros_like(x_flat)
            big_mask = torch.scatter(big_mask, 1, topk_indices, 1)
            self.mask = big_mask.view(*x.shape)
            return x * self.mask
        else:
            return super(UniformRandomTopKModifier, self).forward(x)


class NaiveRandomTopKModifier(TopKModifier):
    def __init__(self, rate: float, input_shape: List[int], random_portion: float = 0.2, batch_dims: int = 1):
        super(NaiveRandomTopKModifier, self).__init__(rate, input_shape, batch_dims)
        self.name = f"nRandTopk{random_portion}"
        self.random_portion = random_portion
        self.non_zero_elements = round(rate * np.prod(input_shape)) or 1
        self.compression_rate = self.non_zero_elements / np.prod(input_shape) * \
                                (1 + np.ceil(np.log2(np.prod(input_shape))) / 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size_flat = np.prod(x.shape[:self.batch_dims])
            x_flat = x.view(batch_size_flat, -1)
            sample_dim = x_flat.shape[1]
            _, top_k_indices = torch.topk(torch.abs(x_flat), self.non_zero_elements, sorted=False)

            probs = torch.ones_like(x_flat) * self.random_portion / (sample_dim - self.non_zero_elements)
            probs = torch.scatter(probs, 1, top_k_indices, (1 - self.random_portion) / self.non_zero_elements)
            selected_indices = torch.multinomial(probs, self.non_zero_elements)
            mask = torch.zeros_like(x_flat)
            self.mask = torch.scatter(mask, 1, selected_indices, 1)
            return self.mask.view(*x.shape) * x

        else:
            return super(NaiveRandomTopKModifier, self).forward(x)


class RandomTopKModifier(TopKModifier):
    def __init__(self, rate: float, input_shape: List[int], exponent: float = 10, batch_dims: int = 1):
        super(RandomTopKModifier, self).__init__(rate, input_shape, batch_dims)
        self.name = f"randTopk{exponent}"
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size_flat = np.prod(x.shape[:self.batch_dims])
            x_flat = x.view(batch_size_flat, -1).abs()
            k = self.non_zero_elements
            x_flat_exponent = (x_flat / (torch.max(x_flat, dim=1, keepdim=True)[0] + 0.001)) ** self.exponent
            big_indices = torch.multinomial(x_flat_exponent, k)
            mask = torch.zeros_like(x_flat)
            mask = torch.scatter(mask, 1, big_indices, 1)
            self.mask = mask.view(*x.shape)
            return x * self.mask
        else:
            return super(RandomTopKModifier, self).forward(x)


class ForwardQuantize(torch.autograd.Function):
    @staticmethod
    def quantize(x: torch.Tensor, n_bits: int):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [batch_size, d]
        quantized_thresholds = (torch.max(x) - torch.min(x)) * \
                                torch.arange(2 ** n_bits + 1, device=x.device) / \
                                (2 ** n_bits) + torch.min(x)
        # [2^N + 1]
        quantized_thresholds = (quantized_thresholds[:-1] + quantized_thresholds[1:]) / 2
        # [2^N]

        diffs = torch.abs(x_flat.unsqueeze(-1) - quantized_thresholds.unsqueeze(-2))  # [batch_size, d, 2^N]
        closest_indices = torch.argmin(diffs, dim=-1)  # [batch_size, d]
        quantized_x = torch.index_select(quantized_thresholds, 0, closest_indices.view(-1))
        return quantized_x.view(x.shape)

    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int):
        ctx.quantize_n_bits = n_bits
        return ForwardQuantize.quantize(x, n_bits)

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        return grad_outputs, None


class TwoWayQuantize(ForwardQuantize):
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        return ForwardQuantize.quantize(grad_outputs, ctx.quantize_n_bits), None


func_forward_quantize = ForwardQuantize.apply
func_two_way_quantize = TwoWayQuantize.apply


class QuantizationModifier(Modifier):
    def __init__(self, n_bits: int, input_shape: List[int], two_way: bool = False):
        if two_way:
            name = "twoWayQuantization"
        else:
            name = "forwardQuantization"
        self.two_way = two_way
        super(QuantizationModifier, self).__init__(name=f"{name}{n_bits}bits")
        self.n_bits = n_bits
        self.compression_rate = n_bits / 32 # + 2**n_bits / np.prod(input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.two_way:
            return func_two_way_quantize(x, self.n_bits)
        else:
            return func_forward_quantize(x, self.n_bits)


if __name__ == '__main__':
    sparsifier = TruncationModifier([20], threshold=5)
    sparsifier.train(False)
    for i in range(10):
        x = torch.arange(20).float()[torch.randperm(20)].view(1, -1)
        print("X:", x)
        compressed = sparsifier(x)
        print("Compressed:", compressed)
