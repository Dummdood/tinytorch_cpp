from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor import Tensor
    from.backward_nodes import Node
from .backward_nodes import AccumulateGrad

class Engine:
    @staticmethod
    def _build_topo(node: Node, visited: set, topo: list):
        if node is None or node in visited:
            return
        visited.add(node)
        for parent in getattr(node, "parents", []):
            if parent is not None:
                Engine._build_topo(parent, visited, topo)
        topo.append(node)

    @staticmethod
    def backward(tensor: Tensor):
        if not tensor.requires_grad:
            return

        data = tensor.data
        grad_output = np.ones_like(data)

        # Leaf tensor case: no grad_fn, accumulate directly
        if tensor.grad_fn is None:
            if tensor.grad is None:
                tensor.grad = grad_output.copy()
            else:
                tensor.grad = tensor.grad + grad_output
            return

        root = tensor.grad_fn
        visited = set()
        topo = []
        Engine._build_topo(root, visited, topo)

        node_grads = {root: grad_output}

        # Process nodes from root toward leaves (reverse post-order)
        for node in reversed(topo):
            g = node_grads.get(node)
            if g is None:
                continue

            # Local backward: get grads for parents
            grads_to_parents = node.apply(g)

            # Propagate to parents
            for parent, pg in zip(node.parents, grads_to_parents):
                if parent is None or pg is None:
                    continue
                node_grads[parent] = node_grads.get(parent, 0) + pg

    @staticmethod
    def print_graph(node, indent=0, visited=None):
        if node is None:
            print(" " * indent + "None")
            return
        if visited is None:
            visited = set()
        if node in visited:
            print(" " * indent + f"{node.__class__.__name__} (visited)")
            return
        visited.add(node)

        label = node.__class__.__name__
        if isinstance(node, AccumulateGrad):
            label += f" -> Tensor(id={id(node.tensor)})"
        print(" " * indent + label)

        for parent in getattr(node, "parents", []):
            Engine.print_graph(parent, indent + 2, visited)