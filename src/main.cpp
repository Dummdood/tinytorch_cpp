#include "tensor.hpp"
#include "loss.hpp"
#include "optim.hpp"
#include "nn.hpp"
#include <iostream>

using namespace autodiff;

int main() {
    // ===============================
    // 1) Linear 2D regression test
    // ===============================
    std::cout << "=== Linear 2D regression test ===\n";

    const int N = 100;
    const int D = 2;

    // Synthetic data: X ~ uniform in [-1, 1]
    Matrix X_data = Matrix::Random(N, D);

    // True parameters
    Matrix W_true(D, 1);
    W_true << 3.0, -2.0;
    double b_true = 0.5;

    // y = X W_true + b_true
    Matrix y_data = X_data * W_true;
    y_data.array() += b_true;

    // Model: Linear(D -> 1)
    Linear lin(D, 1);

    // Optimizer for its parameters
    SGD opt_lin(lin.parameters(), /*lr=*/0.1);

    int epochs_lin = 200;

    for (int epoch = 0; epoch < epochs_lin; ++epoch) {
        double total_loss = 0.0;

        for (int i = 0; i < N; ++i) {
            opt_lin.zero_grad();

            // x_i: (1 x D)
            Matrix x_i_mat(1, D);
            x_i_mat = X_data.row(i);

            // y_i: (1 x 1)
            Matrix y_i_mat(1, 1);
            y_i_mat(0, 0) = y_data(i, 0);

            auto x_i = std::make_shared<Tensor>(x_i_mat, false);
            auto y_i = std::make_shared<Tensor>(y_i_mat, false);

            auto y_pred = lin(x_i);
            auto loss   = mse_loss(y_pred, y_i);

            total_loss += loss->data(0, 0);

            loss->backward();
            opt_lin.step();
        }

        if (epoch % 20 == 0) {
            double avg_loss = total_loss / N;
            std::cout << "epoch " << epoch
                      << " | avg loss = " << avg_loss
                      << " | W = ["
                      << lin.weight->data(0, 0) << ", "
                      << lin.weight->data(1, 0) << "]"
                      << " | b = " << lin.bias->data(0, 0)
                      << "\n";
        }
    }

    std::cout << "True  W = [3, -2], b = 0.5\n";
    std::cout << "Learned W ≈ ["
              << lin.weight->data(0, 0) << ", "
              << lin.weight->data(1, 0) << "], b ≈ "
              << lin.bias->data(0, 0) << "\n\n";


    // ===============================
    // 2) MLP 2D regression test
    // ===============================
    std::cout << "=== MLP 2D regression test ===\n";

    // Tiny MLP: 2 -> 4 -> 1 with ReLU in the hidden layer
    MLP mlp(D, /*hidden_dims=*/{4}, /*output_dim=*/1,
            /*activation=*/"relu",
            /*output_activation=*/"");

    SGD opt_mlp(mlp.parameters(), /*lr=*/0.05);

    int epochs_mlp = 200;

    for (int epoch = 0; epoch < epochs_mlp; ++epoch) {
        double total_loss = 0.0;

        for (int i = 0; i < N; ++i) {
            opt_mlp.zero_grad();

            Matrix x_i_mat(1, D);
            x_i_mat = X_data.row(i);

            Matrix y_i_mat(1, 1);
            y_i_mat(0, 0) = y_data(i, 0);

            auto x_i = std::make_shared<Tensor>(x_i_mat, false);
            auto y_i = std::make_shared<Tensor>(y_i_mat, false);

            auto y_pred = mlp(x_i);
            auto loss   = mse_loss(y_pred, y_i);

            total_loss += loss->data(0, 0);

            loss->backward();
            opt_mlp.step();
        }

        if (epoch % 20 == 0) {
            double avg_loss = total_loss / N;
            std::cout << "epoch " << epoch
                      << " | avg loss = " << avg_loss
                      << "\n";
        }
    }

    std::cout << "First layer W:\n" << mlp.layers[0].weight->data << "\n";

    return 0;
}