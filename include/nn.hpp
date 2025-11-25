#pragma once

#include <vector>
#include <string>
#include <stdexcept>

#include "tensor.hpp"

namespace autodiff {

// -------- Linear layer: y = xW + b (batch_size = 1 for now) --------

struct Linear {
    TensorPtr weight;   // shape: in_features x out_features
    TensorPtr bias;     // shape: 1 x out_features
    bool      has_bias;

    Linear(int in_features, int out_features, bool bias_ = true);

    // Forward pass: x (1 x in_features) -> (1 x out_features)
    TensorPtr operator()(const TensorPtr& x) const;

    // Collect this layer's parameters (weight + optional bias)
    std::vector<TensorPtr> parameters() const;
};

// -------- Simple MLP: stack of Linear layers + activations --------

struct MLP {
    std::vector<Linear> layers;
    std::string         activation;
    std::string         output_activation;

    // hidden_dims can be empty; then this is just a single Linear layer.
    MLP(int input_dim,
        const std::vector<int>& hidden_dims,
        int output_dim,
        const std::string& activation_        = "relu",
        const std::string& output_activation_ = "");

    // Forward: applies all layers and activations
    TensorPtr operator()(const TensorPtr& x) const;

    // Collect all trainable parameters (weights + biases)
    std::vector<TensorPtr> parameters() const;
};

} // namespace autodiff