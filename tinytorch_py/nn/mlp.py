from __future__ import annotations
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..tensor import Tensor

from .module import Module
from .linear import Linear

class MLP(Module):
    def __init__(self, input_dim: int, hidden_dims: int, output_dim: int,
                 activation: str = "relu", output_activation: Optional[str] = None):
        self.layers = []
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(Linear(prev, h))
            prev = h
        self.out_layer = Linear(prev, output_dim)

        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
            if self.activation == "relu":
                out = out.relu()
            elif self.activation == "sigmoid":
                out = out.sigmoid()
        out = self.out_layer(out)
        if self.output_activation == "sigmoid":
            out = out.sigmoid()
        elif self.output_activation == "relu":
            out = out.relu()
        return out