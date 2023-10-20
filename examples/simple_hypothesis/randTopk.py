import numpy as np
import torch
from statistics import NormalDist


from split_learn.modifiers import UniformRandomTopKModifier, RandomTopKModifier, NaiveRandomTopKModifier, TopKModifier


def compute_non_topk_prob(rate: float, exponent: float):
    x = torch.normal(0, 3, [1000, 10000])

    topk = TopKModifier(rate, [10000])(x)
    random_topk = UniformRandomTopKModifier(rate, [10000], exponent)(x)

    prob = torch.mean(((random_topk != topk) & (random_topk != 0)).float()) / rate
    # upper_bound = 2 * exponent / (np.sqrt(2 * np.pi)) * np.exp(-NormalDist(mu=0, sigma=1).inv_cdf(rate)**2/2)
    return prob


if __name__ == '__main__':
    print(compute_non_topk_prob(0.02, 3))