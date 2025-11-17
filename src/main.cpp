#include "tensor.hpp"
#include <iostream>

int main() {
    Tensor x(1, 3, true);
    x(0, 0) = 0.5f;
    x(0, 1) = -1.0f;
    x(0, 2) = 2.0f;

    Tensor y(1, 3);
    y(0, 0) = 1.0f;
    y(0, 1) = 0.0f;
    y(0, 2) = -1.0f;

    Tensor loss = Tensor::mse_loss(x, y);
    std::cout << "loss: " << loss(0, 0) << "\n";

    loss.backward();

    Tensor& g = x.grad();
    std::cout << "grad: "
              << g(0, 0) << " "
              << g(0, 1) << " "
              << g(0, 2) << "\n";
}