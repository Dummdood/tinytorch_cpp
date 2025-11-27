#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Allow root-directory execution: import tinytorch_cpp from ./build
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "build")
sys.path.append(BUILD)

import tinytorch_cpp as tc


def make_dataset(N=100, D=2):
    X = np.random.uniform(-1.0, 1.0, size=(N, D))
    W_true = np.array([[3.0], [-2.0]])
    b_true = 0.5
    y = X @ W_true
    y += b_true
    return X, y, W_true, b_true


def linear_2d_regression_test(X_data, y_data, W_true, b_true):
    print("=== Linear 2D regression test (C++ backend) ===")

    N, D = X_data.shape

    # Linear(D -> 1) with bias
    lin = tc.Linear(D, 1, True)

    # SGD over its parameters (weight + bias)
    opt = tc.SGD(lin.parameters(), 0.1)

    epochs = 200

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(N):
            opt.zero_grad()

            x_i = tc.Tensor(X_data[i : i + 1, :], False)
            y_i = tc.Tensor(y_data[i : i + 1, :], False)

            y_pred = lin(x_i)
            loss = tc.mse_loss(y_pred, y_i)

            total_loss += float(loss.data[0, 0])

            loss.backward()
            opt.step()

        if epoch % 20 == 0:
            W = lin.weight.data
            b = lin.bias.data
            avg_loss = total_loss / N
            print(
                f"epoch {epoch} | avg loss = {avg_loss} "
                f"| W = [{W[0,0]}, {W[1,0]}] | b = {b[0,0]}"
            )

    W = lin.weight.data
    b = lin.bias.data
    print(f"True  W = [{W_true[0,0]}, {W_true[1,0]}], b = {b_true}")
    print(
        f"Learned W ≈ [{W[0,0]}, {W[1,0]}], b ≈ {b[0,0]}\n"
    )


def mlp_2d_regression_test(X_data, y_data):
    print("=== MLP 2D regression test (C++ backend) ===")

    N, D = X_data.shape

    # Tiny MLP: 2 -> 4 -> 1 with ReLU in hidden layer
    mlp = tc.MLP(D, [4], 1, "relu", "")
    opt = tc.SGD(mlp.parameters(), 0.05)

    epochs = 200

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(N):
            opt.zero_grad()

            x_i = tc.Tensor(X_data[i : i + 1, :], False)
            y_i = tc.Tensor(y_data[i : i + 1, :], False)

            y_pred = mlp(x_i)
            loss = tc.mse_loss(y_pred, y_i)

            total_loss += float(loss.data[0, 0])

            loss.backward()
            opt.step()

        if epoch % 20 == 0:
            avg_loss = total_loss / N
            print(f"epoch {epoch} | avg loss = {avg_loss}")

    print("First layer W:")
    print(mlp.layers[0].weight.data)
    print()


if __name__ == "__main__":
    # Match Python TinyTorch behavior: fix RNG for reproducibility
    np.random.seed(0)

    t0 = time.perf_counter()

    X_data, y_data, W_true, b_true = make_dataset()
    linear_2d_regression_test(X_data, y_data, W_true, b_true)
    mlp_2d_regression_test(X_data, y_data)

    t1 = time.perf_counter()
    print(f"[C++ TinyTorch] total seconds = {t1 - t0:.6f}")