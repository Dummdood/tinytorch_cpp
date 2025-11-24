#include "tensor.hpp"
#include "nn.hpp"

#include <iostream>
#include <vector>

using namespace autodiff;

TensorPtr mse_loss(const TensorPtr& y_pred, const TensorPtr& y_true) {
    auto diff = Tensor::sub(y_pred, y_true);  // (1x1)
    auto sq   = Tensor::mul(diff, diff);      // (1x1)
    return sq;
}

int main() {
    std::cout << "=== 2D linear regression with tiny MLP ===\n";

    // Target function: y = 3*x1 - 2*x2
    struct Sample {
        double x1;
        double x2;
        double y;
    };

    std::vector<Sample> data = {
        { 1.0,  0.0,  3.0 },  // 3*1 - 2*0 = 3
        { 0.0,  1.0, -2.0 },  // 3*0 - 2*1 = -2
        { 1.0,  1.0,  1.0 },  // 3*1 - 2*1 = 1
        { 2.0, -1.0,  8.0 }   // 3*2 - 2*(-1) = 8
    };

    // MLP with:
    //   input_dim = 2
    //   hidden_dims = {}  (no hidden layers -> just one Linear)
    //   output_dim = 1
    MLP model(/*input_dim=*/2,
              /*hidden_dims=*/{},
              /*output_dim=*/1,
              /*activation=*/"relu",
              /*output_activation=*/"");   // no final nonlinearity

    // Gather parameters and create SGD optimizer
    auto params = model.parameters();
    SGD optim(params, /*lr=*/0.1);

    const int epochs = 200;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;

        // Stochastic training: loop over samples one by one
        for (const auto& s : data) {
            optim.zero_grad();

            // Build input tensor x: shape (1 x 2)
            Matrix x_mat(1, 2);
            x_mat(0, 0) = s.x1;
            x_mat(0, 1) = s.x2;
            auto x = std::make_shared<Tensor>(x_mat, /*requires_grad=*/false);

            // Build target tensor y_true: shape (1 x 1)
            Matrix y_mat(1, 1);
            y_mat(0, 0) = s.y;
            auto y_true = std::make_shared<Tensor>(y_mat, /*requires_grad=*/false);

            // Forward
            auto y_pred = model(x);            // (1x1)
            auto loss   = mse_loss(y_pred, y_true);

            epoch_loss += loss->data(0, 0);

            // Backward
            loss->backward();

            // Param update
            optim.step();
        }

        if (epoch % 20 == 0) {
            std::cout << "epoch " << epoch
                      << " | avg loss = " << (epoch_loss / data.size())
                      << "\n";
        }
    }

    // Inspect learned weights + bias of the only Linear layer
    auto all_params = model.parameters();
    auto W = all_params[0];  // weight of first (and only) layer
    auto b = all_params[1];  // bias

    std::cout << "\nLearned W (should be close to [3; -2]):\n"
              << W->data << "\n";

    std::cout << "Learned b (should be close to 0):\n"
              << b->data << "\n";

    return 0;
}