import torch
from torch import nn


from deep_utils.convert import Convert


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ViewLayer(nn.Module):
    def __init__(self, shape):
        super(ViewLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def clone_module(m_from: nn.Module, m_to: nn.Module):
    for p_from, p_to in zip(m_from.parameters(), m_to.parameters()):
        p_to.data = Convert.to_tensor(p_from.data.clone().detach())

