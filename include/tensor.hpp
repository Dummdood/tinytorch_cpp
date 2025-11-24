#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "backward_nodes.hpp"

namespace autodiff {

struct Tensor : public std::enable_shared_from_this<Tensor> {
    Matrix data;
    bool   requires_grad;

    Matrix grad;
    bool   grad_initialized;

    NodePtr                        grad_fn;
    std::shared_ptr<AccumulateGrad> accumulate_node;

    // Constructors
    Tensor();
    Tensor(const Matrix& data_,
           bool          requires_grad_ = false,
           NodePtr       grad_fn_       = nullptr);

    // Basic info
    bool is_leaf() const;
    std::pair<int,int> shape() const;

    // Backprop entry point
    void backward();

private:
    // ---- Shape checks ----
    static void check_binary_op_shapes(const Matrix& a,
                                       const Matrix& b,
                                       const std::string& op_name);

    // Build the parent function for a tensor (either its grad_fn or AccumulateGrad)
    static NodePtr make_parent_fn(const TensorPtr& t);

    // ---- Generic helpers for ops (templates must stay in header) ----
    template <typename OpBackward, typename F>
    static TensorPtr binary_op(const TensorPtr& a,
                               const TensorPtr& b,
                               const std::string& op_name,
                               F op);

    template <typename OpBackward, typename F>
    static TensorPtr unary_op(const TensorPtr& a, F op);

public:
    // ---- Binary op helpers ----
    static TensorPtr add   (const TensorPtr& a, const TensorPtr& b);
    static TensorPtr sub   (const TensorPtr& a, const TensorPtr& b);
    static TensorPtr mul   (const TensorPtr& a, const TensorPtr& b);
    static TensorPtr div   (const TensorPtr& a, const TensorPtr& b);
    static TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
    static TensorPtr pow   (const TensorPtr& a, const TensorPtr& b);

    // ---- Unary op helpers ----
    static TensorPtr relu   (const TensorPtr& a);
    static TensorPtr sigmoid(const TensorPtr& a);
    static TensorPtr exp    (const TensorPtr& a);
    static TensorPtr log    (const TensorPtr& a);
    static TensorPtr sum    (const TensorPtr& a);
    static TensorPtr mean   (const TensorPtr& a);
};

// ===== Template / inline definitions that must stay in the header =====

// ---- Generic helpers ----

template <typename OpBackward, typename F>
TensorPtr Tensor::binary_op(const TensorPtr& a,
                            const TensorPtr& b,
                            const std::string& op_name,
                            F op) {
    if (!a || !b) {
        throw std::runtime_error(op_name + ": null TensorPtr");
    }

    check_binary_op_shapes(a->data, b->data, op_name);

    Matrix out_data = op(a->data, b->data);
    bool requires_grad = a->requires_grad || b->requires_grad;

    if (!requires_grad) {
        return std::make_shared<Tensor>(out_data, false);
    }

    auto ctx = std::make_shared<Context>();
    ctx->save_for_backward(a, b);

    auto node = std::make_shared<OpBackward>();
    node->ctx = ctx;

    NodePtr parent_a = make_parent_fn(a);
    NodePtr parent_b = make_parent_fn(b);

    node->parents.clear();
    node->parents.resize(2);

    if (parent_a) node->parents[0] = parent_a;
    if (parent_b) node->parents[1] = parent_b;

    auto out = std::make_shared<Tensor>(out_data, true, node);
    return out;
}

template <typename OpBackward, typename F>
TensorPtr Tensor::unary_op(const TensorPtr& a, F op) {
    if (!a) {
        throw std::runtime_error("unary_op: null TensorPtr");
    }

    Matrix out_data = op(a->data);

    if (!a->requires_grad) {
        return std::make_shared<Tensor>(out_data, false);
    }

    auto ctx = std::make_shared<Context>();
    ctx->save_for_backward(a);

    auto node = std::make_shared<OpBackward>();
    node->ctx = ctx;

    NodePtr parent = make_parent_fn(a);
    node->parents.clear();
    node->parents.resize(1);
    if (parent) node->parents[0] = parent;

    auto out = std::make_shared<Tensor>(out_data, true, node);
    return out;
}

// ---- Binary op helpers ----

inline TensorPtr Tensor::add(const TensorPtr& a, const TensorPtr& b) {
    return binary_op<AddBackward>(
        a, b, "add",
        [](const Matrix& x, const Matrix& y) { return x + y; }
    );
}

inline TensorPtr Tensor::sub(const TensorPtr& a, const TensorPtr& b) {
    return binary_op<SubBackward>(
        a, b, "sub",
        [](const Matrix& x, const Matrix& y) { return x - y; }
    );
}

inline TensorPtr Tensor::mul(const TensorPtr& a, const TensorPtr& b) {
    return binary_op<MulBackward>(
        a, b, "mul",
        [](const Matrix& x, const Matrix& y) {
            return (x.array() * y.array()).matrix();
        }
    );
}

inline TensorPtr Tensor::div(const TensorPtr& a, const TensorPtr& b) {
    return binary_op<DivBackward>(
        a, b, "div",
        [](const Matrix& x, const Matrix& y) {
            return (x.array() / y.array()).matrix();
        }
    );
}

inline TensorPtr Tensor::matmul(const TensorPtr& a, const TensorPtr& b) {
    return binary_op<MatMulBackward>(
        a, b, "matmul",
        [](const Matrix& x, const Matrix& y) { return x * y; }
    );
}

inline TensorPtr Tensor::pow(const TensorPtr& a, const TensorPtr& b) {
    return binary_op<PowBackward>(
        a, b, "pow",
        [](const Matrix& base, const Matrix& exponent) {
            Matrix out = base;
            out = base.array().pow(exponent.array());
            return out;
        }
    );
}

// ---- Unary op helpers ----

inline TensorPtr Tensor::relu(const TensorPtr& a) {
    return unary_op<ReluBackward>(
        a,
        [](const Matrix& x) {
            Matrix out = x;
            out = x.cwiseMax(0.0);
            return out;
        }
    );
}

inline TensorPtr Tensor::sigmoid(const TensorPtr& a) {
    return unary_op<SigmoidBackward>(
        a,
        [](const Matrix& x) {
            Matrix out = x;
            out = (1.0 / (1.0 + (-x.array()).exp())).matrix();
            return out;
        }
    );
}

inline TensorPtr Tensor::exp(const TensorPtr& a) {
    return unary_op<ExpBackward>(
        a,
        [](const Matrix& x) {
            Matrix out = x;
            out = x.array().exp().matrix();
            return out;
        }
    );
}

inline TensorPtr Tensor::log(const TensorPtr& a) {
    return unary_op<LogBackward>(
        a,
        [](const Matrix& x) {
            Matrix out = x;
            out = x.array().log().matrix();
            return out;
        }
    );
}

inline TensorPtr Tensor::sum(const TensorPtr& a) {
    return unary_op<SumBackward>(
        a,
        [](const Matrix& x) {
            Matrix out(1, 1);
            out(0, 0) = x.sum();
            return out;
        }
    );
}

inline TensorPtr Tensor::mean(const TensorPtr& a) {
    return unary_op<MeanBackward>(
        a,
        [](const Matrix& x) {
            Matrix out(1, 1);
            out(0, 0) = x.mean();
            return out;
        }
    );
}

// ---- Free operator overloads ----

inline TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) {
    return Tensor::add(a, b);
}

inline TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) {
    return Tensor::sub(a, b);
}

inline TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) {
    return Tensor::mul(a, b);
}

inline TensorPtr operator/(const TensorPtr& a, const TensorPtr& b) {
    return Tensor::div(a, b);
}

inline TensorPtr operator%(const TensorPtr& a, const TensorPtr& b) {
    return Tensor::matmul(a, b);
}

} // namespace autodiff