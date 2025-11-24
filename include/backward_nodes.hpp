// backward_nodes.hpp
#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <stdexcept>
#include <utility>
#include <cmath>

#include <Eigen/Dense>

namespace autodiff {

struct Tensor;      // forward
struct AccumulateGrad;

using Matrix    = Eigen::MatrixXd;
using TensorPtr = std::shared_ptr<Tensor>;
using NodePtr   = std::shared_ptr<struct Node>;

struct Context {
    std::vector<TensorPtr> saved_tensors;

    void save_for_backward(const std::vector<TensorPtr>& tensors);

    template <typename... Ts>
    void save_for_backward(const Ts&... tensors) {
        saved_tensors.clear();
        (saved_tensors.emplace_back(tensors), ...);
    }
};

struct Node : public std::enable_shared_from_this<Node> {
    std::vector<std::weak_ptr<Node>> parents;
    std::shared_ptr<Context> ctx;

    virtual ~Node() = default;

    virtual std::vector<std::optional<Matrix>>
    apply(const Matrix& grad_output) = 0;
};

// ------ Binary Ops ------

struct BinaryOpBackward : public Node {
    std::vector<std::optional<Matrix>>
    apply(const Matrix& grad_output) override;

protected:
    virtual std::pair<std::optional<Matrix>, std::optional<Matrix>>
    compute_grads(const TensorPtr& a,
                  const TensorPtr& b,
                  const Matrix&    grad_output) = 0;
};

struct AddBackward : public BinaryOpBackward {
protected:
    std::pair<std::optional<Matrix>, std::optional<Matrix>>
    compute_grads(const TensorPtr& a,
                  const TensorPtr& b,
                  const Matrix&    grad_output) override;
};

struct SubBackward : public BinaryOpBackward {
protected:
    std::pair<std::optional<Matrix>, std::optional<Matrix>>
    compute_grads(const TensorPtr& a,
                  const TensorPtr& b,
                  const Matrix&    grad_output) override;
};

struct MulBackward : public BinaryOpBackward {
protected:
    std::pair<std::optional<Matrix>, std::optional<Matrix>>
    compute_grads(const TensorPtr& a,
                  const TensorPtr& b,
                  const Matrix&    grad_output) override;
};

struct DivBackward : public BinaryOpBackward {
protected:
    std::pair<std::optional<Matrix>, std::optional<Matrix>>
    compute_grads(const TensorPtr& a,
                  const TensorPtr& b,
                  const Matrix&    grad_output) override;
};

struct MatMulBackward : public BinaryOpBackward {
protected:
    std::pair<std::optional<Matrix>, std::optional<Matrix>>
    compute_grads(const TensorPtr& a,
                  const TensorPtr& b,
                  const Matrix&    grad_output) override;
};

struct PowBackward : public BinaryOpBackward {
protected:
    std::pair<std::optional<Matrix>, std::optional<Matrix>>
    compute_grads(const TensorPtr& a,
                  const TensorPtr& b,
                  const Matrix&    grad_output) override;
};

// ------ Unary Ops ------

struct UnaryOpBackward : public Node {
    std::vector<std::optional<Matrix>>
    apply(const Matrix& grad_output) override;

protected:
    virtual std::optional<Matrix>
    compute_grad(const TensorPtr& a,
                 const Matrix&    grad_output) = 0;
};

struct ReluBackward : public UnaryOpBackward {
protected:
    std::optional<Matrix>
    compute_grad(const TensorPtr& a,
                 const Matrix&    grad_output) override;
};

struct SigmoidBackward : public UnaryOpBackward {
protected:
    std::optional<Matrix>
    compute_grad(const TensorPtr& a,
                 const Matrix&    grad_output) override;
};

struct ExpBackward : public UnaryOpBackward {
protected:
    std::optional<Matrix>
    compute_grad(const TensorPtr& a,
                 const Matrix&    grad_output) override;
};

struct LogBackward : public UnaryOpBackward {
protected:
    std::optional<Matrix>
    compute_grad(const TensorPtr& a,
                 const Matrix&    grad_output) override;
};

struct SumBackward : public UnaryOpBackward {
protected:
    std::optional<Matrix>
    compute_grad(const TensorPtr& a,
                 const Matrix&    grad_output) override;
};

struct MeanBackward : public UnaryOpBackward {
protected:
    std::optional<Matrix>
    compute_grad(const TensorPtr& a,
                 const Matrix&    grad_output) override;
};

// ------ Leaf Node ------

struct AccumulateGrad : public Node {
    std::weak_ptr<Tensor> tensor;

    explicit AccumulateGrad(const TensorPtr& t);  // <-- declaration only

    std::vector<std::optional<Matrix>>
    apply(const Matrix& grad_output) override;
};


} // namespace autodiff