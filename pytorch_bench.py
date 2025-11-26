#!/usr/bin/env python3
import time
import torch


def make_dataset(N=100, D=2, seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Match semantics: uniform-ish in [-1, 1]
    X = 2.0 * torch.rand((N, D), generator=g, device=device, dtype=dtype) - 1.0

    W_true = torch.tensor([[3.0], [-2.0]], device=device, dtype=dtype)
    b_true = 0.5

    y = X @ W_true + b_true
    return X, y, W_true, b_true


def linear_2d_regression_test(X, y, W_true, b_true, device="cpu", dtype=torch.float64):
    print("=== Linear 2D regression test (PyTorch) ===")

    N, D = X.shape

    # Model params: W (D x 1), b (1 x 1)
    g = torch.Generator(device=device)
    g.manual_seed(0)
    W = torch.randn((D, 1), generator=g, device=device, dtype=dtype, requires_grad=True)
    b = torch.zeros((1, 1), device=device, dtype=dtype, requires_grad=True)

    lr = 0.1
    epochs = 200

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(N):
            # zero_grad
            if W.grad is not None: W.grad.zero_()
            if b.grad is not None: b.grad.zero_()

            x_i = X[i:i+1, :]          # (1 x D)
            y_i = y[i:i+1, :]          # (1 x 1)

            y_pred = x_i @ W + b       # (1 x 1)
            loss = torch.mean((y_pred - y_i) ** 2)

            total_loss += loss.item()

            loss.backward()

            with torch.no_grad():
                W -= lr * W.grad
                b -= lr * b.grad

        if epoch % 20 == 0:
            avg_loss = total_loss / N
            print(
                f"epoch {epoch} | avg loss = {avg_loss} | W = [{W[0,0].item()}, {W[1,0].item()}] | b = {b[0,0].item()}"
            )

    print(f"True  W = [{W_true[0,0].item()}, {W_true[1,0].item()}], b = {b_true}")
    print(
        f"Learned W ≈ [{W[0,0].item()}, {W[1,0].item()}], b ≈ {b[0,0].item()}\n"
    )


def mlp_2d_regression_test(X, y, device="cpu", dtype=torch.float64):
    print("=== MLP 2D regression test (PyTorch) ===")

    N, D = X.shape

    # Tiny MLP: 2 -> 4 -> 1 with ReLU
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(D, 4, bias=True, dtype=dtype),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1, bias=True, dtype=dtype),
    ).to(device)

    lr = 0.05
    epochs = 200

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(N):
            # zero_grad
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            x_i = X[i:i+1, :]
            y_i = y[i:i+1, :]

            y_pred = model(x_i)
            loss = torch.mean((y_pred - y_i) ** 2)
            total_loss += loss.item()

            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad

        if epoch % 20 == 0:
            avg_loss = total_loss / N
            print(f"epoch {epoch} | avg loss = {avg_loss}")

    # Print first layer weight like your C++ / TinyTorch scripts
    first_layer_W = model[0].weight.detach().cpu()
    print("First layer W:")
    print(first_layer_W)


def main():
    device = "cpu"
    dtype = torch.float64

    t0 = time.perf_counter()

    X, y, W_true, b_true = make_dataset(N=100, D=2, seed=0, device=device, dtype=dtype)
    linear_2d_regression_test(X, y, W_true, b_true, device=device, dtype=dtype)
    mlp_2d_regression_test(X, y, device=device, dtype=dtype)

    t1 = time.perf_counter()
    print(f"\n[PyTorch] total seconds = {t1 - t0}")


if __name__ == "__main__":
    main()