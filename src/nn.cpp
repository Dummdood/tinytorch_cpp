#include "nn.hpp"

namespace autodiff {

// ===== Linear =====

Linear::Linear(int in_features, int out_features, bool bias_)
    : has_bias(bias_)
{
    // Random init in [-1, 1]
    Matrix w = Matrix::Random(in_features, out_features);
    weight = std::make_shared<Tensor>(w, /*requires_grad=*/true);

    if (has_bias) {
        Matrix b = Matrix::Zero(1, out_features);
        bias = std::make_shared<Tensor>(b, /*requires_grad=*/true);
    }
}

TensorPtr Linear::operator()(const TensorPtr& x) const {
    if (!x) {
        throw std::runtime_error("Linear::operator(): x is null");
    }

    // x: (1 x in_features), weight: (in_features x out_features)
    auto out = Tensor::matmul(x, weight);  // (1 x out_features)

    if (has_bias) {
        // For now we *don't* support broadcasting.
        // So we enforce batch_size == 1 and exact shape match.
        if (out->data.rows() != bias->data.rows() ||
            out->data.cols() != bias->data.cols()) {
            throw std::runtime_error(
                "Linear: bias shape must match output shape exactly "
                "(no broadcasting yet)"
            );
        }
        out = Tensor::add(out, bias);
    }

    return out;
}

// ===== MLP =====

MLP::MLP(int input_dim,
         const std::vector<int>& hidden_dims,
         int output_dim,
         const std::string& activation_,
         const std::string& output_activation_)
    : activation(activation_)
    , output_activation(output_activation_)
{
    int prev = input_dim;

    // Hidden layers
    for (int h : hidden_dims) {
        layers.emplace_back(prev, h, /*bias=*/true);
        prev = h;
    }

    // Output layer
    layers.emplace_back(prev, output_dim, /*bias=*/true);
}

TensorPtr MLP::operator()(const TensorPtr& x) const {
    TensorPtr out = x;

    // All layers except last use "activation"
    if (layers.size() >= 2) {
        for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
            out = layers[i](out);
            if (activation == "relu") {
                out = Tensor::relu(out);
            } else if (activation == "sigmoid") {
                out = Tensor::sigmoid(out);
            }
        }
    }

    // Final layer
    out = layers.back()(out);
    if (output_activation == "relu") {
        out = Tensor::relu(out);
    } else if (output_activation == "sigmoid") {
        out = Tensor::sigmoid(out);
    }

    return out;
}

std::vector<TensorPtr> MLP::parameters() const {
    std::vector<TensorPtr> params;
    params.reserve(layers.size() * 2);

    for (const auto& layer : layers) {
        params.push_back(layer.weight);
        if (layer.has_bias) {
            params.push_back(layer.bias);
        }
    }
    return params;
}

// ===== SGD =====

SGD::SGD(const std::vector<TensorPtr>& params_, double lr_)
    : params(params_)
    , lr(lr_)
{}

void SGD::zero_grad() {
    for (auto& p : params) {
        if (!p) continue;
        if (p->grad_initialized) {
            p->grad.setZero();
            p->grad_initialized = false;
        }
    }
}

void SGD::step() {
    for (auto& p : params) {
        if (!p) continue;
        if (p->requires_grad && p->grad_initialized) {
            p->data -= lr * p->grad;
        }
    }
}

} // namespace autodiff