#pragma once

#include "module.hpp"
#include "tensor.hpp"
#include <random>
#include <cmath>

// // Fully-connected layer: y = x W + b
class Linear : public Module {
    int in_features_;
    int out_features_;
    Tensor W_;  // shape: (in_features, out_features)
    Tensor b_;  // shape: (1, out_features)

    void init_params() {
        // // Xavier uniform init
        float fan_in  = static_cast<float>(in_features_);
        float fan_out = static_cast<float>(out_features_);
        float limit   = std::sqrt(6.0f / (fan_in + fan_out));

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (int i = 0; i < in_features_; ++i) {
            for (int j = 0; j < out_features_; ++j) {
                W_(i, j) = dist(gen);
            }
        }
        for (int j = 0; j < out_features_; ++j) {
            b_(0, j) = 0.0f;
        }
    }

public:
    Linear(int in_features, int out_features)
        : in_features_(in_features),
          out_features_(out_features),
          W_(in_features, out_features, /*requires_grad=*/true),
          b_(1, out_features, /*requires_grad=*/true)
    {
        init_params();
    }

    // // Forward: y = x W + b
    Tensor forward(const Tensor& x) override {
        Tensor y = Tensor::matmul(x, W_);     // (batch × out_features)

        // bias add: broadcast b_ across batch dimension
        int batch = y.rows();
        int out_f = y.cols();
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < out_f; ++j) {
                y(i, j) = y(i, j) + b_(0, j);
            }
        }
        return y;
    }

    // // Parameters: weights and bias
    std::vector<Tensor*> parameters() override {
        return { &W_, &b_ };
    }
};