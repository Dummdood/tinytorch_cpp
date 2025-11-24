#include "tensor.hpp"
#include "loss.hpp"
#include "optim.hpp"
#include <iostream>

using namespace autodiff;

int main() {
    // ===============================
    // 1) Tiny 1D training demo
    // ===============================
    std::cout << "=== Tiny 1D training demo ===\n";

    Matrix x_data(1, 1);
    x_data(0, 0) = 2.0;

    Matrix w_data(1, 1);
    w_data(0, 0) = 1.0;

    auto x = std::make_shared<Tensor>(x_data, false);  // no grad
    auto w = std::make_shared<Tensor>(w_data, true);   // learnable

    Matrix y_true_data(1, 1);
    y_true_data(0, 0) = 8.0;
    auto y_true = std::make_shared<Tensor>(y_true_data, false);

    double lr1 = 0.1;

    for (int step = 0; step < 10; ++step) {
        w->zero_grad();

        // y_pred = x * w
        auto y_pred = x * w;            // uses operator* -> Tensor::mul
        auto loss   = mse_loss(y_pred, y_true);

        loss->backward();

        w->data = w->data - lr1 * w->grad;

        std::cout << "step " << step
                  << " | loss = " << loss->data(0, 0)
                  << " | w = " << w->data(0, 0)
                  << " | w.grad = " << w->grad(0, 0)
                  << "\n";
    }

    std::cout << "Final w (1D demo) ≈ " << w->data(0, 0) << "\n\n";


    // =========================================
    // 2) 2D "stress test" regression with many ops
    // =========================================
    std::cout << "=== 2D stress test (matmul + exp + log + sigmoid + pow) ===\n";

    const int N = 100;  // number of samples
    const int D = 2;    // feature dimension

    // Synthetic data X ~ uniform-ish in [-1, 1]
    Matrix X_data = Matrix::Random(N, D);

    // True weights for underlying function
    Matrix W_true(D, 1);
    W_true << 3.0, -2.0;

    // True logits: X * W_true
    Matrix logits_true = X_data * W_true;

    // True probabilities via sigmoid(logits_true)
    Matrix probs_true = (1.0 / (1.0 + (-logits_true.array()).exp())).matrix();

    // We'll train the model to match probs_true^2 using a more complex forward:
    // logits -> exp -> log -> sigmoid -> pow(·, 2)
    Matrix y_true_sq_data = probs_true.array().square().matrix();

    // Wrap data in Tensors (no grad)
    auto X_stress      = std::make_shared<Tensor>(X_data, false);
    auto y_true_sq     = std::make_shared<Tensor>(y_true_sq_data, false);

    // Learnable weight vector W (D x 1), random init
    Matrix W_init = 0.1 * Matrix::Random(D, 1);
    auto W = std::make_shared<Tensor>(W_init, true);

    // Exponent tensor (same shape as output): all 2.0
    Matrix exp_mat = Matrix::Constant(N, 1, 2.0);
    auto exponent = std::make_shared<Tensor>(exp_mat, false);  // no grad

    double lr2     = 0.1;
    int    epochs  = 15000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        W->zero_grad();

        // Forward:
        // logits = X * W                               (N x 1)
        auto logits = Tensor::matmul(X_stress, W);

        // exp_logits = exp(logits)                    (N x 1)
        auto exp_logits = Tensor::exp(logits);

        // log_exp_logits = log(exp_logits)            (N x 1) ≈ logits
        auto log_exp_logits = Tensor::log(exp_logits);

        // probs = sigmoid(log_exp_logits)             (N x 1)
        auto probs = Tensor::sigmoid(log_exp_logits);

        // probs_sq = probs^2 via pow(probs, exponent) (N x 1)
        auto probs_sq = Tensor::pow(probs, exponent);

        // Loss: MSE(probs_sq, y_true_sq)
        auto loss = mse_loss(probs_sq, y_true_sq);

        // Backward through the whole graph
        loss->backward();

        // Gradient step on W
        W->data = W->data - lr2 * W->grad;

        if (epoch % 50 == 0) {
            std::cout << "epoch " << epoch
                      << " | loss = " << loss->data(0, 0)
                      << " | W = ["
                      << W->data(0, 0) << ", " << W->data(1, 0) << "]\n";
        }
    }

    std::cout << "\nTrue W = [3, -2]\n";
    std::cout << "Learned W ≈ ["
              << W->data(0, 0) << ", " << W->data(1, 0) << "]\n";

    return 0;
}