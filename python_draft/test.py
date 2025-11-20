import numpy as np
import operator

class Context:
    def __init__(self):
        self.saved_tensors: list["Tensor"] = []

    def save_for_backward(self, *tensors: "Tensor"):
        self.saved_tensors = tensors
 
class Node:
    def __init__(self):
        self.next_functions = []
        self.ctx = None

    def backward(self, grad_output):
        raise NotImplementedError

class MulBackward(Node):
    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        grad_a = grad_output * b.data if a.requires_grad else None
        grad_b = grad_output * a.data if b.requires_grad else None

        if grad_a is not None:
            self.next_functions[0].backward(grad_a)
        if grad_b is not None:
            self.next_functions[1].backward(grad_b)

class AddBackward(Node):
    def backward(self, grad_output):    
        a, b = self.ctx.saved_tensors
        grad_a = grad_output if a.requires_grad else None
        grad_b = grad_output if b.requires_grad else None

        if grad_a is not None:
            self.next_functions[0].backward(grad_a)
        if grad_b is not None:
            self.next_functions[1].backward(grad_b)
    
class AccumulateGrad(Node):
    def __init__(self, tensor: "Tensor"):
        super().__init__()
        self.tensor = tensor

    def backward(self, grad_output):
        if self.tensor.grad is None:
            self.tensor.grad = grad_output
        else:
            self.tensor.grad += grad_output



class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None, grad=None):
        self.data = data
        self.requires_grad: bool = requires_grad
        self.grad_fn = grad_fn
        self.grad: float = grad
        self._accumulate_node = None
    @property
    def is_leaf(self):
        return self.grad_fn is None and self.requires_grad 
    
    @staticmethod
    def __make_parent_fn(t: "Tensor"):
        if t.is_leaf:
            t._accumulate_node = t._accumulate_node if t._accumulate_node else AccumulateGrad(t)
            return t.get_accumulate_node()
        elif t.requires_grad and t.grad_fn is not None:
            return t.grad_fn
        else:
            return None
        
    def get_accumulate_node(self):
        if self._accumulate_node is None:
            self._accumulate_node = AccumulateGrad(self)
        return self._accumulate_node
        
    def __operation(self, t2: "Tensor", OperationBackward: type["Node"], op):
        if not isinstance(t2, Tensor):
            raise TypeError
        
        out = op(self.data, t2.data)

        requires_grad = self.requires_grad or t2.requires_grad
        if not requires_grad:
            return Tensor(data=out, requires_grad=False)
        
        ctx = Context()
        ctx.save_for_backward(self, t2)

        node = OperationBackward()
        node.ctx = ctx
        
        parent_a = Tensor.__make_parent_fn(self)
        parent_b = Tensor.__make_parent_fn(t2)
        node.next_functions = [parent_a, parent_b]

        out = Tensor(out, requires_grad=True, grad_fn=node)
        return out
    
    def __add__(self, other: "Tensor"):
        return self.__operation(other, AddBackward, operator.add)
    
    def __mul__(self, multiplier: "Tensor"):
        return self.__operation(multiplier, MulBackward, operator.mul)