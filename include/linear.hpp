#include "tensor.hpp"

class Linear {
    int in_features_;
    int out_features_;
    Tensor W_;
    Tensor b_;
public:
    // Constructors
    explicit Linear(int in_features, int out_features)
        : in_features_(in_features),
        out_features_(out_features),
        W_(in_features, out_features),
        b_(1, out_features)
    {}

    Tensor forward(const Tensor& t_in) const {
        return Tensor::add(Tensor::matmul(t_in, W_), b_);
    }
};