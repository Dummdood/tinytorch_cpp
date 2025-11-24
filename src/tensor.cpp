// tensor.cpp
#include "tensor.hpp"
#include <unordered_set>
#include <unordered_map>

namespace autodiff {

namespace {

// DFS to build a topological order of nodes reachable from `node`
void build_topo(const NodePtr& node,
                std::unordered_set<Node*>& visited,
                std::vector<NodePtr>& topo) {
    if (!node) return;

    Node* raw = node.get();
    if (visited.count(raw)) {
        return;
    }
    visited.insert(raw);

    // Recurse on parents (the graph edges for backward)
    for (auto& wparent : node->parents) {
        if (auto parent = wparent.lock()) {
            build_topo(parent, visited, topo);
        }
    }

    topo.push_back(node);
}

} // anonymous namespace

void Tensor::backward() {
    if (!requires_grad) {
        return;
    }

    // Seed gradient at this tensor: d(loss)/d(this)
    Matrix grad_out = Matrix::Ones(data.rows(), data.cols());

    // If this is a leaf (no grad_fn), just accumulate into grad and stop
    if (!grad_fn) {
        if (!grad_initialized) {
            grad = grad_out;
            grad_initialized = true;
        } else {
            grad += grad_out;
        }
        return;
    }

    // Build topological order of nodes in the backward graph
    std::unordered_set<Node*> visited;
    std::vector<NodePtr> topo;
    build_topo(grad_fn, visited, topo);

    // Map from raw Node* to its accumulated gradient
    std::unordered_map<Node*, Matrix> node_grads;
    node_grads[grad_fn.get()] = grad_out;

    // Process nodes from root (this->grad_fn) toward leaves
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        NodePtr node = *it;
        Node* raw = node.get();

        auto itg = node_grads.find(raw);
        if (itg == node_grads.end()) {
            continue; // no gradient flowed to this node
        }

        const Matrix& g = itg->second;

        // Local backward: each Node returns a grad (optional) per parent
        auto grads_to_parents = node->apply(g);

        // Push grads down to parents
        for (std::size_t i = 0; i < node->parents.size(); ++i) {
            auto& wparent = node->parents[i];
            if (wparent.expired()) {
                continue;
            }
            if (i >= grads_to_parents.size()) {
                continue;
            }
            auto& opt_pg = grads_to_parents[i];
            if (!opt_pg.has_value()) {
                continue; // parent didn't require grad
            }

            NodePtr parent = wparent.lock();
            Node* parent_raw = parent.get();

            auto itp = node_grads.find(parent_raw);
            if (itp == node_grads.end()) {
                node_grads[parent_raw] = *opt_pg;
            } else {
                itp->second += *opt_pg; // multiple paths accumulate
            }
        }
    }
}

} // namespace autodiff
