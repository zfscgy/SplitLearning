from pathlib import Path
import matplotlib.pyplot as plt


import torch

from deep_utils.convert import Convert

from split_learn.modifiers import Modifier, RandomTopKModifier, NaiveRandomTopKModifier, UniformRandomTopKModifier


export_dir = "images"
if not Path(export_dir).is_dir():
    Path(export_dir).mkdir()


feature_size = 1000
batch_size = 100000

def plot_distribution(modifier: Modifier):
    random_batch = Convert.to_tensor(torch.abs(torch.normal(0, 1, [batch_size, feature_size])))
    random_batch, _ = torch.sort(random_batch, dim=1)
    with torch.no_grad():
        modifier(random_batch)
    distribution = modifier.mask.sum(dim=0) / batch_size
    distribution = Convert.to_numpy(distribution)
    plt.plot(distribution, label=modifier.name)



if __name__ == '__main__':
    for rate in [0.02, 0.05, 0.1]:
        for modifier in [NaiveRandomTopKModifier(rate, [1000]),
                         RandomTopKModifier(rate, [1000], 3),
                         UniformRandomTopKModifier(rate, [1000], 2)]:
            plot_distribution(modifier)
        plt.vlines(1000 - modifier.non_zero_elements, 0, 1, colors='red')
        plt.legend()
        plt.show()
