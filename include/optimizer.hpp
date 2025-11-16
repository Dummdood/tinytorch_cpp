#pragma once

#include <vector>
#include "tensor.hpp"
#include "module.hpp"

// // Stochastic Gradient Descent optimizer
class SGD {
    float lr_;
public:
    explicit SGD(float lr)
        : lr_(lr) {}

    // // One optimization step on a list of parameters
    void step(const std::vector<Tensor*>& params) {
        for (Tensor* p : params) {
            if (!p) continue;
            if (!p->requires_grad()) continue;
            if (!p->has_grad()) continue;   // nothing to update yet

            Tensor& g = p->grad();          // gradient tensor
            int R = p->rows();
            int C = p->cols();

            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    (*p)(i, j) -= lr_ * g(i, j);
                }
            }
        }
    }

    // // Convenience: step directly on a Module
    void step(Module& m) {
        auto params = m.parameters();
        step(params);
    }
};