#include "loss.hpp"

namespace autodiff {

TensorPtr mse_loss(const TensorPtr& y_pred, const TensorPtr& y_true) {
    if (!y_pred || !y_true) {
        throw std::runtime_error("mse_loss: null tensor");
    }

    auto [r1, c1] = y_pred->shape();
    auto [r2, c2] = y_true->shape();
    if (r1 != r2 || c1 != c2) {
        throw std::runtime_error("mse_loss: shape mismatch between y_pred and y_true");
    }

    auto diff = Tensor::sub(y_pred, y_true);
    auto sq   = Tensor::mul(diff, diff);
    auto mse  = Tensor::mean(sq);

    return mse;
}

} // namespace autodiff