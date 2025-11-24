#include "tensor.hpp"
#include "loss.hpp"
#include <iostream>

using namespace autodiff;

int main() {
    std::cout << "=== Tiny training demo ===\n";

    Matrix x_data(1, 1);
    x_data(0, 0) = 2.0;

    Matrix w_data(1, 1);
    w_data(0, 0) = 1.0;

    auto x = std::make_shared<Tensor>(x_data, false);
    auto w = std::make_shared<Tensor>(w_data, true);

    Matrix y_true_data(1, 1);
    y_true_data(0, 0) = 8.0;
    auto y_true = std::make_shared<Tensor>(y_true_data, false);

    double lr = 0.1;

    for (int step = 0; step < 10; ++step) {
        // ergonomic zero_grad
        w->zero_grad();

        // forward
        // auto y_pred = Tensor::mul(x, w);
        auto y_pred = x * w;
        auto loss   = mse_loss(y_pred, y_true);

        // backward
        loss->backward();

        // SGD step
        w->data = w->data - lr * w->grad;

        std::cout << "step " << step
                  << " | loss = " << loss->data(0, 0)
                  << " | w = " << w->data(0, 0)
                  << " | w.grad = " << w->grad(0, 0)
                  << "\n";
    }

    return 0;
}