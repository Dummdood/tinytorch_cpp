#pragma once

#include <vector>
#include "tensor.hpp"
#include <memory>

// Base class for all trainable modules (layers)
class Module {
public:
    virtual ~Module() = default;

    // Forward pass: every module must implement this
    virtual Tensor forward(const Tensor& x) = 0;

    // Parameters of this module, we override in subclasses with params
    virtual std::vector<Tensor*> parameters() {
        return {};
    }

    // Zero gradients of all parameters
    void zero_grad() {
        auto params = parameters();
        for (Tensor* p : params) {
            if (p) {
                p->zero_grad();
            }
        }
    }
};

// Container that applies submodules in sequence: y = m_n(...m_2(m_1(x)))
class Sequential : public Module {
    std::vector<std::shared_ptr<Module>> modules_;
public:
    Sequential() = default;

    // Add a submodule to the sequence
    void add(std::shared_ptr<Module> m) {
        modules_.push_back(std::move(m));
    }

    // Forward: pipe input through all modules
    Tensor forward(const Tensor& x) override {
        Tensor out = x;
        for (auto& m : modules_) {
            out = m->forward(out);
        }
        return out;
    }

    // Parameters: flatten parameters from all submodules
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> all;
        for (auto& m : modules_) {
            auto sub_params = m->parameters();
            all.insert(all.end(), sub_params.begin(), sub_params.end());
        }
        return all;
    }
};

class ReLU : public Module {
public:
    ReLU() = default;

    // Forward: just call Tensor::relu
    Tensor forward(const Tensor& x) override {
        return Tensor::relu(x);
    }

    // No parameters
    std::vector<Tensor*> parameters() override {
        return {};
    }
};

class Sigmoid : public Module {
public:
    Sigmoid() = default;

    Tensor forward(const Tensor& x) override {
        return Tensor::sigmoid(x);
    }

    std::vector<Tensor*> parameters() override {
        return {};
    }
};