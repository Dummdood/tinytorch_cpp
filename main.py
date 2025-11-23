import numpy as np

from tinytorch_py.tensor import Tensor
from tinytorch_py.optim.sgd import SGD
from tinytorch_py.nn.mlp import MLP


def simple_test():
    np.random.seed(0)

    X = Tensor(np.array([[1.0, 2.0]]), requires_grad=False)
    W = Tensor(np.random.randn(2, 1), requires_grad=True)
    b = Tensor(np.array([[0.0]]), requires_grad=True)
    y_true = Tensor(np.array([[1.0]]), requires_grad=False)

    opt = SGD([W, b], lr=0.1)

    for step in range(5):
        opt.zero_grad()

        z = X @ W
        z = z + b
        y_pred = z.sigmoid()

        diff = y_pred - y_true
        loss = diff * diff

        loss.backward()

        print(
            f"[simple] step {step} | loss = {loss.data} "
            f"| W.grad = {W.grad} | b.grad = {b.grad}"
        )

        opt.step()


def mlp_test():
    np.random.seed(0)

    model = MLP(
        input_dim=2,
        hidden_dims=[4, 4],
        output_dim=1,
        activation="relu",
        output_activation="sigmoid",
    )

    optimizer = SGD(model.parameters(), lr=0.1)

    x = Tensor(np.array([[1.0, -1.0]]), requires_grad=False)
    y_true = Tensor(np.array([[1.0]]), requires_grad=False)

    for step in range(10):
        optimizer.zero_grad()

        y_pred = model(x)
        diff = y_pred - y_true
        loss = diff * diff

        loss.backward()

        print(f"[mlp] step {step} | loss = {loss.data}")

        optimizer.step()


if __name__ == "__main__":
    simple_test()
    mlp_test()