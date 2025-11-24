import numpy as np
from .tensor import Tensor


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Mean squared error over all elements.
    """
    diff = y_pred - y_true
    return (diff * diff).mean()


def binary_cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Binary cross-entropy:
        -[ y * log(p) + (1-y) * log(1-p) ]
    averaged over all elements.
    Assumes y_pred in (0,1). 
    """
    one = Tensor(np.ones_like(y_pred.data), requires_grad=False)
    loss = - (y_true * y_pred.log() + (one - y_true) * (one - y_pred).log())
    return loss.mean()