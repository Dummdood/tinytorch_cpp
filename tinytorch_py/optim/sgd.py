from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from tensor import Tensor

class SGD:
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data = p.data - self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = 0