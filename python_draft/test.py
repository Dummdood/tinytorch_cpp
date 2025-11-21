import numpy as np

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
    
    
class BinaryOpBackward(Node):
    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        grad_a, grad_b = self._compute_grads(a, b, grad_output)

        if grad_a is not None and self.next_functions[0] is not None:
            self.next_functions[0].backward(grad_a)
        if grad_b is not None and self.next_functions[1] is not None:
            self.next_functions[1].backward(grad_b)

    def _compute_grads(self, a, b, grad_output):
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
    
    
class UnaryOpBackward(Node):
    def backward(self, grad_output):
        (a,) = self.ctx.saved_tensors
        grad_a = self._compute_grad(a, grad_output)
        if grad_a is not None and self.next_functions[0] is not None:
            self.next_functions[0].backward(grad_a)

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
            return t._accumulate_node
        elif t.requires_grad and t.grad_fn is not None:
            return t.grad_fn
        else:
            return None
    
    def __increment_grad(self, grad_output: float):
        if self.grad is None:
            self.grad = grad_output
        else:
            self.grad += grad_output
        
    def backward(self, grad_output=1.0):
        if not self.requires_grad:
            return

        if self.grad_fn is None:
            self.__increment_grad(grad_output)
            return

        self.grad_fn.backward(grad_output)
        
    def __operation(self, t2: "Tensor", OpBackward: type["Node"], op):
        if not isinstance(t2, Tensor):
            t2 = Tensor(t2, requires_grad=False)
        
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
        node.next_functions = [parent_a, parent_b]

        out = Tensor(out, requires_grad=True, grad_fn=node)
        return out
    
    def __unary_op(self, OpBackward: type["Node"], fn):
        out = fn(self.data)
        if not self.requires_grad:
            return Tensor(out, requires_grad=False)

        ctx = Context()
        ctx.save_for_backward(self)

        node = OpBackward()
        node.ctx = ctx

        parent = Tensor.__make_parent_fn(self)
        node.next_functions = [parent]

        return Tensor(out, requires_grad=True, grad_fn=node)
    
    def __add__(self, other: "Tensor"):
        return self.__operation(other, AddBackward, lambda a, b: a + b)
    
    def __sub__(self, other: "Tensor"):
        return self.__operation(other, SubBackward, lambda a, b: a - b)
    
    def __mul__(self, multiplier: "Tensor"):
        return self.__operation(multiplier, MulBackward, lambda a, b: a * b)

    def __truediv__(self, other: "Tensor"):
        return self.__operation(other, DivBackward, lambda a, b: a / b)
    
    def __matmul__(self, other: "Tensor"):
        return self.__operation(other, MatMulBackward, lambda a, b: a @ b)
    
    def relu(self):
        return self.__unary_op(ReluBackward, lambda x: np.maximum(0, x))

    def sigmoid(self):
        return self.__unary_op(SigmoidBackward, lambda x: 1.0 / (1.0 + np.exp(-x)))

class SGD:
    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
            p.data = p.data - self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Module:
    def parameters(self):
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                yield attr
            elif isinstance(attr, Module):
                yield from attr.parameters()
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        yield from item.parameters()
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield item

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        # He-ish init (scaled by sqrt(2 / in_features)), but simple is fine too
        w_data = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        self.weight = Tensor(w_data, requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class MLP(Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation="relu", output_activation=None):
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

def mlp_test():
    np.random.seed(0)

    # 2 -> [4, 4] -> 1, ReLU hidden, sigmoid output
    model = MLP(input_dim=2,
                hidden_dims=[4, 4],
                output_dim=1,
                activation="relu",
                output_activation="sigmoid")

    optimizer = SGD(model.parameters(), lr=0.1)

    # Single training sample (keeps loss scalar-like: shape (1,1))
    x = Tensor(np.array([[1.0, -1.0]]), requires_grad=False)
    y_true = Tensor(np.array([[1.0]]), requires_grad=False)

    for step in range(500):
        optimizer.zero_grad()

        y_pred = model(x)
        diff = y_pred - y_true
        loss = diff * diff

        loss.backward()

        print(
            f"[mlp] step {step} | loss = {loss.data} "
        )

        optimizer.step()


if __name__ == "__main__":
    mlp_test()