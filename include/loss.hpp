#pragma once

#include "tensor.hpp"

namespace autodiff {

// mse(y_pred, y_true) = mean((y_pred - y_true)^2)
TensorPtr mse_loss(const TensorPtr& y_pred, const TensorPtr& y_true);

}
