#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Allow root-directory execution: import tinytorch_cpp from ./build
ROOT_DIR = os.path.dirname(__file__)
BUILD_DIR = os.path.join(ROOT_DIR, "build")
sys.path.append(BUILD_DIR)

import tinytorch_cpp as tc


def make_dataset(N=100, D=2):
    X = np.random.uniform(-1.0, 1.0, size=(N, D))
    W_true = np.array([[3.0], [-2.0]])
    b_true = 0.5
    y = X @ W_true
    y += b_true
    return X, y, W_true, b_true


def tiny_1d_demo():
    print("=== Tiny 1D training demo (C++ backend) ===")

    x_data = np.array([[2.0]], dtype=np.float64)
    w_data = np.array([[1.0]], dtype=np.float64)
    y_true_data = np.array([[8.0]], dtype=np.float64)

    x = tc.Tensor(x_data, False)
    w = tc.Tensor(w_data, True)
    y_true = tc.Tensor(y_true_data, False)

    lr = 0.1

    for step in range(10):
        w.zero_grad()

        # y_pred = x * w
        y_pred = x * w
        loss = tc.mse_loss(y_pred, y_true)
        loss.backward()

        # SGD update on w
        w.data = w.data - lr * w.grad

        loss_val = float(loss.data[0, 0])
        w_val = float(w.data[0, 0])
        g_val = float(w.grad[0, 0])

        # Match the style of your previous prints as much as possible
        print(f"step {step} | loss = {loss_val} | w = {w_val} | w.grad = {g_val}")

    print(f"Final w (1D demo) ≈ {float(w.data[0, 0])}\n")


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
    print("Using module:", tc)

    # Match Python TinyTorch behavior: fix RNG for reproducibility
    np.random.seed(0)

    t0 = time.perf_counter()

    tiny_1d_demo()

    X_data, y_data, W_true, b_true = make_dataset()
    linear_2d_regression_test(X_data, y_data, W_true, b_true)
    mlp_2d_regression_test(X_data, y_data)

    t1 = time.perf_counter()
    print(f"[C++ TinyTorch] total seconds = {t1 - t0:.6f}")