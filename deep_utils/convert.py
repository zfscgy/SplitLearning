from typing import Union, List

import numpy as np
import torch
from torch import nn


from deep_utils.config import GlobalConfig


class Convert:
    @staticmethod
    def to_numpy(x: torch.Tensor):
        try:
            return x.cpu().detach().numpy()
        except:
            pass

        return x.cpu().numpy()

    @staticmethod
    def to_tensor(x: Union[List, torch.Tensor, np.ndarray, float]):
        if isinstance(x, List):
            return [Convert.to_tensor(e) for e in x]
        if isinstance(x, torch.Tensor):
            return x.to(GlobalConfig.device)
        else:
            return torch.tensor(x).to(GlobalConfig.device)

    @staticmethod
    def model_to_device(m: nn.Module):
        return m.to(GlobalConfig.device)

    @staticmethod
    def model_to_cpu(m_cuda: nn.Module):
        return m_cuda.to('cpu')
