import time
import numpy as np

from tinytorch_py.tensor import Tensor
from tinytorch_py.optim.sgd import SGD
from tinytorch_py.nn.mlp import MLP
from tinytorch_py.nn.linear import Linear
from tinytorch_py.losses import mse_loss


def tiny_1d_demo_py():
    print("=== Tiny 1D training demo (Python) ===")

    x = Tensor(np.array([[2.0]]), requires_grad=False)
    w = Tensor(np.array([[1.0]]), requires_grad=True)
    y_true = Tensor(np.array([[8.0]]), requires_grad=False)

    lr = 0.1
    opt = SGD([w], lr=lr)

    for step in range(10):
        opt.zero_grad()

        y_pred = x * w
        loss = mse_loss(y_pred, y_true)

        loss.backward()
        opt.step()

        print(
            f"step {step} | loss = {loss.data.item()} | w = {w.data.item()} | w.grad = {w.grad.item()}"
        )

    print(f"Final w (1D demo) ≈ {w.data.item()}\n")


def linear_2d_regression_test_py():
    print("=== Linear 2D regression test (Python) ===")

    N = 100
    D = 2

    X_data = np.random.uniform(-1.0, 1.0, size=(N, D))

    W_true = np.array([[3.0], [-2.0]])
    b_true = 0.5
    y_data = X_data @ W_true + b_true

    lin = Linear(D, 1, bias=True)
    opt = SGD(lin.parameters(), lr=0.1)

    epochs = 200
    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(N):
            opt.zero_grad()

            x_i = Tensor(X_data[i:i+1, :], requires_grad=False)
            y_i = Tensor(y_data[i:i+1, :], requires_grad=False)

            y_pred = lin(x_i)
            loss = mse_loss(y_pred, y_i)

            total_loss += loss.data.item()

            loss.backward()
            opt.step()

        if epoch % 20 == 0:
            avg_loss = total_loss / N
            W0 = lin.weight.data[0, 0].item()
            W1 = lin.weight.data[1, 0].item()
            b0 = lin.bias.data[0, 0].item()
            print(f"epoch {epoch} | avg loss = {avg_loss} | W = [{W0}, {W1}] | b = {b0}")

    print("True  W = [3, -2], b = 0.5")
    W0 = lin.weight.data[0, 0].item()
    W1 = lin.weight.data[1, 0].item()
    b0 = lin.bias.data[0, 0].item()
    print(f"Learned W ≈ [{W0}, {W1}], b ≈ {b0}\n")


def mlp_2d_regression_test_py():
    print("=== MLP 2D regression test (Python) ===")

    N = 100
    D = 2

    X_data = np.random.uniform(-1.0, 1.0, size=(N, D))

    W_true = np.array([[3.0], [-2.0]])
    b_true = 0.5
    y_data = X_data @ W_true + b_true

    mlp = MLP(
        input_dim=D,
        hidden_dims=[4],
        output_dim=1,
        activation="relu",
        output_activation="",
    )
    opt = SGD(mlp.parameters(), lr=0.05)

    epochs = 200
    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(N):
            opt.zero_grad()

            x_i = Tensor(X_data[i:i+1, :], requires_grad=False)
            y_i = Tensor(y_data[i:i+1, :], requires_grad=False)

            y_pred = mlp(x_i)
            loss = mse_loss(y_pred, y_i)

            total_loss += loss.data.item()

            loss.backward()
            opt.step()

        if epoch % 20 == 0:
            avg_loss = total_loss / N
            print(f"epoch {epoch} | avg loss = {avg_loss}")

    print("First layer W:\n" + str(mlp.layers[0].weight.data) + "\n")


def main():
    np.random.seed(0)

    t0 = time.perf_counter()
    tiny_1d_demo_py()
    linear_2d_regression_test_py()
    mlp_2d_regression_test_py()
    t1 = time.perf_counter()

    print(f"[Python TinyTorch] total seconds = {t1 - t0:.6f}")


if __name__ == "__main__":
    main()