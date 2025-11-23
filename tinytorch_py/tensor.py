
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .backward_nodes import Node
from .engine import Engine
from .backward_nodes import (
    Context,
    # Binary Ops
    AddBackward, SubBackward, MulBackward,
    DivBackward, MatMulBackward, PowBackward,
    # Unary Ops
    ReluBackward, SigmoidBackward, ExpBackward, 
    LogBackward, SumBackward, MeanBackward,
    # Leaf Node Op
    AccumulateGrad
)

ALLOWED_DTYPES = {
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64
}

class Tensor:
    def __init__(self, data, requires_grad:bool=False, grad_fn:Optional[Node]=None, grad=None):
        self.data:Optional["np.ndarray"] = self.to_2d_array(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = grad
        self._accumulate_node = None

    @property
    def is_leaf(self):
        return self.grad_fn is None and self.requires_grad
    
    @staticmethod
    def to_2d_array(x):
        arr = np.asarray(x)

        # Check dtype is allowed
        if arr.dtype.type not in ALLOWED_DTYPES:
            raise TypeError(f"Unsupported dtype: {arr.dtype}. "
                            f"Allowed dtypes: {sorted([t.__name__ for t in ALLOWED_DTYPES])}")

        # Scalar (0-D) → reshape to (1,1)
        if arr.ndim == 0:
            return arr.reshape(1, 1)

        # Vector (1-D) → reshape to (1, n)
        elif arr.ndim == 1:
            return arr.reshape(1, -1)

        # Matrix (2-D) → leave as is
        elif arr.ndim == 2:
            return arr

        else:
            raise ValueError("Input must be scalar, 1D, or 2D.")
        
    def backward(self):
        Engine.backward(self)

    def print_graph(self):
        print(f"Tensor(data={self.data}, requires_grad={self.requires_grad}, is_leaf={self.is_leaf})")
        if self.grad_fn is None:
            print("  grad_fn: None")
            return
        print(f"  grad_fn: {self.grad_fn.__class__.__name__}")
        print("Computation graph (backward):")
        Engine.print_graph(self.grad_fn, indent=2)

    @staticmethod
    def _check_binary_op_shapes(a: "Tensor", b: "Tensor", op_name: str):
        a_shape = a.shape
        b_shape = b.shape

        # Strict elementwise ops
        if op_name in ("add", "sub", "mul", "div", "pow"):
            if a_shape != b_shape:
                raise ValueError(
                    f"Shape mismatch in {op_name}: "
                    f"{a_shape} and {b_shape} must match exactly"
                )
            return

        # Matmul rule
        if op_name == "matmul":
            if a_shape[-1] != b_shape[-2]:
                raise ValueError(
                    f"Matmul shape mismatch: {a_shape} @ {b_shape} is invalid"
                )
            return

        raise ValueError(f"Unknown op: {op_name}")

    @staticmethod
    def __make_parent_fn(t: "Tensor"):
        if t.is_leaf:
            t._accumulate_node = t._accumulate_node if t._accumulate_node else AccumulateGrad(t)
            return t._accumulate_node
        elif t.requires_grad and t.grad_fn is not None:
            return t.grad_fn
        else:
            return None
        
    def __binary_op(
        self,
        t2: "Tensor",
        OpBackward: Type["Node"],
        op: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> "Tensor":
        if not isinstance(t2, Tensor):
            raise TypeError(f"{OpBackward.__name__.replace("Backward", "").lower()} "
                             "can only be performed on floats")

        op_name = OpBackward.__name__.replace("Backward", "").lower()
        self._check_binary_op_shapes(self.data, t2.data, op_name)
        
        out = op(self.data, t2.data)

        requires_grad = self.requires_grad or t2.requires_grad
        if not requires_grad:
            return Tensor(data=out, requires_grad=False)
        
        ctx = Context()
        ctx.save_for_backward(self, t2)

        node = OpBackward()
        node.ctx = ctx
        
        parent_a = Tensor.__make_parent_fn(self)
        parent_b = Tensor.__make_parent_fn(t2)
        node.parents = [parent_a, parent_b]

        out = Tensor(out, requires_grad=True, grad_fn=node)
        return out
    
    def __unary_op(self, OpBackward: type["Node"], op: Callable[[np.ndarray], np.ndarray]):
        out = op(self.data)
        if not self.requires_grad:
            return Tensor(out, requires_grad=False)

        ctx = Context()
        ctx.save_for_backward(self)

        node = OpBackward()
        node.ctx = ctx

        parent = Tensor.__make_parent_fn(self)
        node.parents = [parent]

        return Tensor(out, requires_grad=True, grad_fn=node)
    
    # ------ Binary Ops ------

    def __add__(self, other: "Tensor"):
        return self.__binary_op(other, AddBackward, lambda a, b: a + b)
    
    def __sub__(self, other: "Tensor"):
        return self.__binary_op(other, SubBackward, lambda a, b: a - b)
    
    def __mul__(self, multiplier: "Tensor"):
        return self.__binary_op(multiplier, MulBackward, lambda a, b: a * b)

    def __truediv__(self, other: "Tensor"):
        return self.__binary_op(other, DivBackward, lambda a, b: a / b)
    
    def __matmul__(self, other: "Tensor"):
        return self.__binary_op(other, MatMulBackward, lambda a, b: a @ b)
    
    def __pow__(self, exponent: "Tensor") -> "Tensor":
        return self.__binary_op(exponent, PowBackward, lambda a, b: a ** b)
    
    # ------ Unary Ops ------
    
    def relu(self):
        return self.__unary_op(ReluBackward, lambda x: np.maximum(0, x))

    def sigmoid(self):
        return self.__unary_op(SigmoidBackward, lambda x: 1.0 / (1.0 + np.exp(-x)))
    
    def exp(self) -> "Tensor":
        return self.__unary_op(ExpBackward, lambda x: np.exp(x))

    def log(self) -> "Tensor":
        return self.__unary_op(LogBackward, lambda x: np.log(x))
    
    def sum(self) -> "Tensor":
        return self.__unary_op(SumBackward, lambda x: np.array([[x.sum()]]))

    def mean(self) -> "Tensor":
        return self.__unary_op(MeanBackward, lambda x: np.array([[x.mean()]]))