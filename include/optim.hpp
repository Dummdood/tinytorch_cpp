#pragma once

#include <vector>
#include "tensor.hpp"

namespace autodiff {

struct SGD {
    std::vector<TensorPtr> params;
    double lr;

    SGD(const std::vector<TensorPtr>& ps, double lr_)
        : params(ps), lr(lr_) {}

    void zero_grad() {
        for (auto& p : params) {
            if (!p) continue;
            if (p->grad_initialized) {
                p->grad.setZero();
                p->grad_initialized = false;
            }
        }
    }

    void step() {
        for (auto& p : params) {
            if (!p) continue;
            if (p->grad_initialized) {
                p->data -= lr * p->grad;
            }
        }
    }
};

} // namespace autodiff