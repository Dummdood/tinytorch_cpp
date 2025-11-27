// backward_nodes.cpp
#include "tensor.hpp"

namespace autodiff {

void Context::save_for_backward(const std::vector<TensorPtr>& tensors) {
    saved_tensors.clear();
    saved_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        saved_tensors.emplace_back(t);
    }
}

std::vector<std::optional<Matrix>>
BinaryOpBackward::apply(const Matrix& grad_output) {
    if (!ctx || ctx->saved_tensors.size() != 2) {
        throw std::runtime_error("BinaryOpBackward: context not properly initialized");
    }

    auto a = ctx->saved_tensors[0];
    auto b = ctx->saved_tensors[1];

    if (!a || !b) {
        throw std::runtime_error("BinaryOpBackward: saved tensors are null");
    }

    auto [grad_a, grad_b] = compute_grads(a, b, grad_output);
    return { std::move(grad_a), std::move(grad_b) };
}

// === AddBackward ===
std::pair<std::optional<Matrix>, std::optional<Matrix>>
AddBackward::compute_grads(const TensorPtr& a,
                           const TensorPtr& b,
                           const Matrix&    grad_output) {
    std::optional<Matrix> grad_a;
    std::optional<Matrix> grad_b;

    if (a->requires_grad) grad_a = grad_output;
    if (b->requires_grad) grad_b = grad_output;

    return { std::move(grad_a), std::move(grad_b) };
}

// === SubBackward ===
std::pair<std::optional<Matrix>, std::optional<Matrix>>
SubBackward::compute_grads(const TensorPtr& a,
                           const TensorPtr& b,
                           const Matrix&    grad_output) {
    std::optional<Matrix> grad_a;
    std::optional<Matrix> grad_b;

    if (a->requires_grad) grad_a = grad_output;
    if (b->requires_grad) grad_b = -grad_output;

    return { std::move(grad_a), std::move(grad_b) };
}

// === MulBackward ===
std::pair<std::optional<Matrix>, std::optional<Matrix>>
MulBackward::compute_grads(const TensorPtr& a,
                           const TensorPtr& b,
                           const Matrix&    grad_output) {
    std::optional<Matrix> grad_a;
    std::optional<Matrix> grad_b;

    if (a->requires_grad) {
        grad_a = grad_output.array() * b->data.array();
    }
    if (b->requires_grad) {
        grad_b = grad_output.array() * a->data.array();
    }
    return { std::move(grad_a), std::move(grad_b) };
}

// === DivBackward ===
std::pair<std::optional<Matrix>, std::optional<Matrix>>
DivBackward::compute_grads(const TensorPtr& a,
                           const TensorPtr& b,
                           const Matrix&    grad_output) {
    std::optional<Matrix> grad_a;
    std::optional<Matrix> grad_b;

    if (a->requires_grad) {
        grad_a = grad_output.array() / b->data.array();
    }
    if (b->requires_grad) {
        Matrix denom = b->data.array().square();
        grad_b = -grad_output.array() * a->data.array() / denom.array();
    }
    return { std::move(grad_a), std::move(grad_b) };
}

// === MatMulBackward ===
std::pair<std::optional<Matrix>, std::optional<Matrix>>
MatMulBackward::compute_grads(const TensorPtr& a,
                              const TensorPtr& b,
                              const Matrix&    grad_output) {
    std::optional<Matrix> grad_a;
    std::optional<Matrix> grad_b;

    if (a->requires_grad) {
        grad_a = grad_output * b->data.transpose();
    }
    if (b->requires_grad) {
        grad_b = a->data.transpose() * grad_output;
    }
    return { std::move(grad_a), std::move(grad_b) };
}

// === PowBackward ===
std::pair<std::optional<Matrix>, std::optional<Matrix>>
PowBackward::compute_grads(const TensorPtr& a,
                           const TensorPtr& b,
                           const Matrix&    grad_output) {
    const Matrix& base     = a->data;
    const Matrix& exponent = b->data;

    std::optional<Matrix> grad_a;
    std::optional<Matrix> grad_b;

    if (a->requires_grad) {
        Matrix base_pow = base.array().pow(exponent.array() - 1.0);
        grad_a = grad_output.array() * exponent.array() * base_pow.array();
    }

    if (b->requires_grad) {
        Matrix base_pow = base.array().pow(exponent.array());
        Matrix log_base = base.array().log();
        grad_b = grad_output.array() * base_pow.array() * log_base.array();
    }

    return { std::move(grad_a), std::move(grad_b) };
}

// === UnaryOps ===
std::vector<std::optional<Matrix>>
UnaryOpBackward::apply(const Matrix& grad_output) {
    if (!ctx || ctx->saved_tensors.size() != 1) {
        throw std::runtime_error("UnaryOpBackward: context not properly initialized");
    }

    auto a = ctx->saved_tensors[0];
    if (!a) {
        throw std::runtime_error("UnaryOpBackward: saved tensor is null");
    }

    auto grad_a = compute_grad(a, grad_output);
    return { std::move(grad_a) };
}

std::optional<Matrix>
ReluBackward::compute_grad(const TensorPtr& a,
                           const Matrix&    grad_output) {
    if (!a->requires_grad) return std::nullopt;
    Matrix mask = (a->data.array() > 0.0).cast<double>();
    return grad_output.array() * mask.array();
}

std::optional<Matrix>
SigmoidBackward::compute_grad(const TensorPtr& a,
                              const Matrix&    grad_output) {
    if (!a->requires_grad) return std::nullopt;
    Matrix sig = 1.0 / (1.0 + (-a->data.array()).exp());
    return grad_output.array() * sig.array() * (1.0 - sig.array());
}

std::optional<Matrix>
ExpBackward::compute_grad(const TensorPtr& a,
                          const Matrix&    grad_output) {
    if (!a->requires_grad) return std::nullopt;
    Matrix ex = a->data.array().exp();
    return grad_output.array() * ex.array();
}

std::optional<Matrix>
LogBackward::compute_grad(const TensorPtr& a,
                          const Matrix&    grad_output) {
    if (!a->requires_grad) return std::nullopt;
    Matrix inv = 1.0 / a->data.array();
    return grad_output.array() * inv.array();
}

std::optional<Matrix>
SumBackward::compute_grad(const TensorPtr& a,
                          const Matrix&    grad_output) {
    if (!a->requires_grad) return std::nullopt;
    double g = grad_output(0, 0);
    Matrix grad = Matrix::Constant(a->data.rows(), a->data.cols(), g);
    return grad;
}

std::optional<Matrix>
MeanBackward::compute_grad(const TensorPtr& a,
                           const Matrix&    grad_output) {
    if (!a->requires_grad) return std::nullopt;
    double g = grad_output(0, 0);
    double N = static_cast<double>(a->data.size());
    Matrix grad = Matrix::Constant(a->data.rows(), a->data.cols(), g / N);
    return grad;
}

// === AccumulateGrad ===
AccumulateGrad::AccumulateGrad(const TensorPtr& t)
    : tensor(t) {}

std::vector<std::optional<Matrix>>
AccumulateGrad::apply(const Matrix& grad_output) {
    auto t = tensor.lock();
    if (!t) {
        return {};
    }

    if (!t->grad_initialized) {
        t->grad = grad_output;
        t->grad_initialized = true;
    } else {
        t->grad += grad_output;
    }

    return {};
}


}
