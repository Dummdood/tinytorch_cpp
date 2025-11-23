import numpy as np

from .module import Module
from ..tensor import Tensor

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        w_data = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        self.weight = Tensor(w_data, requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out