from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from tensor import Tensor

class Context:
    def __init__(self):
        self.saved_tensors: list["Tensor"] = []

    def save_for_backward(self, *tensors: "Tensor"):
        self.saved_tensors = tensors
 

class Node:
    def __init__(self):
        self.parents: Node = []
        self.ctx: Context = None

    def apply(self, grad_output: "np.ndarray"):
        raise NotImplementedError
    
    
# ------ Binary Ops ------
    
class BinaryOpBackward(Node):
    def apply(self, 
              grad_output: "np.ndarray"
              ) -> tuple[Optional["np.ndarray"], Optional["np.ndarray"]]:
        a, b = self.ctx.saved_tensors
        return self._compute_grads(a, b, grad_output)

    def _compute_grads(
        self,
        a: "Tensor",
        b: "Tensor",
        grad_output: "np.ndarray"
    ) -> tuple[Optional["np.ndarray"], Optional["np.ndarray"]]:
        raise NotImplementedError

class AddBackward(BinaryOpBackward):
    def _compute_grads(self, a, b, grad_output):
        grad_a = grad_output if a.requires_grad else None
        grad_b = grad_output if b.requires_grad else None
        return grad_a, grad_b

class SubBackward(BinaryOpBackward):
    def _compute_grads(self, a, b, grad_output):
        grad_a = grad_output if a.requires_grad else None
        grad_b = -grad_output if b.requires_grad else None
        return grad_a, grad_b

class MulBackward(BinaryOpBackward):
    def _compute_grads(self, a, b, grad_output):
        grad_a = grad_output * b.data if a.requires_grad else None
        grad_b = grad_output * a.data if b.requires_grad else None
        return grad_a, grad_b
    
class DivBackward(BinaryOpBackward):
    def _compute_grads(self, a, b, grad_output):
        grad_a = grad_output / b.data if a.requires_grad else None
        grad_b = -grad_output * a.data / (b.data ** 2) if b.requires_grad else None
        return grad_a, grad_b
    
class MatMulBackward(BinaryOpBackward):
    def _compute_grads(self, a, b, grad_output):
        grad_a = grad_output @ b.data.T if a.requires_grad else None
        grad_b = a.data.T @ grad_output if b.requires_grad else None
        return grad_a, grad_b

class PowBackward(BinaryOpBackward):
    def _compute_grads(self, a: "Tensor", b: "Tensor", grad_output: np.ndarray):
        base = a.data
        exponent = b.data

        grad_a = None
        grad_b = None

        # d/d base (base**exponent) = exponent * base**(exponent - 1)
        if a.requires_grad:
            grad_a = grad_output * exponent * (base ** (exponent - 1))

        # d/d exponent (base**exponent) = base**exponent * log(base)
        # (undefined for base <= 0)
        if b.requires_grad:
            grad_b = grad_output * (base ** exponent) * np.log(base)

        return grad_a, grad_b

# ------ Unary Ops ------

class UnaryOpBackward(Node):
    def apply(self, grad_output):
        (a,) = self.ctx.saved_tensors
        grad_a = self._compute_grad(a, grad_output)
        return (grad_a,)

    def _compute_grad(self, a, grad_output):
        raise NotImplementedError
    
class ReluBackward(UnaryOpBackward):
    def _compute_grad(self, a, grad_output):
        if not a.requires_grad:
            return None
        return grad_output * (a.data > 0)

class SigmoidBackward(UnaryOpBackward):
    def _compute_grad(self, a, grad_output):
        if not a.requires_grad:
            return None
        # sigmoid(x) * (1 - sigmoid(x))
        sig = 1.0 / (1.0 + np.exp(-a.data))
        return grad_output * sig * (1.0 - sig)
    
class ExpBackward(UnaryOpBackward):
    def _compute_grad(self, a: "Tensor", grad_output: np.ndarray):
        if not a.requires_grad:
            return None
        # d/dx exp(x) = exp(x)
        return grad_output * np.exp(a.data)

class LogBackward(UnaryOpBackward):
    def _compute_grad(self, a: "Tensor", grad_output: np.ndarray):
        if not a.requires_grad:
            return None
        # d/dx log(x) = 1/x
        return grad_output * (1.0 / a.data)

class SumBackward(UnaryOpBackward):
    def _compute_grad(self, a: "Tensor", grad_output: np.ndarray):
        if not a.requires_grad:
            return None
        # y = sum(x_ij); dy/dx_ij = 1
        return grad_output * np.ones_like(a.data)

class MeanBackward(UnaryOpBackward):
    def _compute_grad(self, a: "Tensor", grad_output: np.ndarray):
        if not a.requires_grad:
            return None
        # y = (1/N) * sum(x_ij); dy/dx_ij = 1/N
        N = a.data.size
        return grad_output * (np.ones_like(a.data) / N)


# ------ Leaf Node ------

class AccumulateGrad(Node):
    def __init__(self, tensor: "Tensor"):
        super().__init__()
        self.tensor = tensor

    def apply(self, grad_output):
        self.tensor.grad += grad_output
        return ()