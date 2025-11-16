#pragma once
#include <vector>
#include <functional>
#include <memory>

class Tensor;

struct Node {
    std::vector<Tensor*> parents;
    std::function<void(const Tensor&)> backward;

    Node() = default;
    Node(std::vector<Tensor*> parents,
         std::function<void(const Tensor&)> backward)
        : parents(std::move(parents)),
          backward(std::move(backward)) {}
};
