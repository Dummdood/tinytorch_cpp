import numpy as np
from .tensor import Tensor


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    diff = y_pred - y_true
    return (diff * diff).mean()


def binary_cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    one = Tensor(np.ones_like(y_pred.data), requires_grad=False)
    loss = - (y_true * y_pred.log() + (one - y_true) * (one - y_pred).log())
    return loss.mean()