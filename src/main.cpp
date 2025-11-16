#include <iostream>
#include "tensor.hpp"
#include "linear.hpp"

int main() {
    // // Create a Linear layer: 3 inputs → 2 outputs
    // Linear layer(3, 2);

    // // Create an input vector of shape (1 × 3)
    // Tensor x(1, 3);
    // x(0, 0) = 0.5f;
    // x(0, 1) = -1.0f;
    // x(0, 2) = 2.0f;

    // // Forward pass
    // Tensor y = layer.forward(x);

    // // Apply softmax for classification
    // Tensor probs = Tensor::softmax(y);

    // // Print results
    // std::cout << "Output logits:" << std::endl;
    // for (int j = 0; j < y.cols(); j++) {
    //     std::cout << y(0, j) << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Softmax probabilities:" << std::endl;
    // for (int j = 0; j < probs.cols(); j++) {
    //     std::cout << probs(0, j) << " ";
    // }
    // std::cout << std::endl;

    return 0;
}