#include "tensor.hpp"

#include <unordered_set>
#include <unordered_map>

namespace autodiff {

// ---- Constructors ----

Tensor::Tensor()
    : data()
    , requires_grad(false)
    , grad()
    , grad_initialized(false)
    , grad_fn(nullptr)
    , accumulate_node(nullptr)
{}

Tensor::Tensor(const Matrix& data_,
               bool          requires_grad_,
               NodePtr       grad_fn_)
    : data(data_)
    , requires_grad(requires_grad_)
    , grad(Matrix::Zero(data_.rows(), data_.cols()))
    , grad_initialized(false)
    , grad_fn(std::move(grad_fn_))
    , accumulate_node(nullptr)
{}

// ---- Basic info ----

bool Tensor::is_leaf() const {
    return !grad_fn && requires_grad;
}

std::pair<int,int> Tensor::shape() const {
    return { static_cast<int>(data.rows()),
             static_cast<int>(data.cols()) };
}

void Tensor::zero_grad() {
    grad.resize(data.rows(), data.cols());
    grad.setZero();
    grad_initialized = false;
}

// ---- Shape checks ----

void Tensor::check_binary_op_shapes(const Matrix& a,
                                    const Matrix& b,
                                    const std::string& op_name) {
    auto a_rows = a.rows();
    auto a_cols = a.cols();
    auto b_rows = b.rows();
    auto b_cols = b.cols();

    if (op_name == "add" || op_name == "sub" ||
        op_name == "mul" || op_name == "div" ||
        op_name == "pow") {

        if (a_rows != b_rows || a_cols != b_cols) {
            throw std::runtime_error(
                "Shape mismatch in " + op_name +
                ": matrices must match exactly");
        }
        return;
    }

    if (op_name == "matmul") {
        if (a_cols != b_rows) {
            throw std::runtime_error(
                "Matmul shape mismatch: (" +
                std::to_string(a_rows) + "," + std::to_string(a_cols) +
                ") @ (" +
                std::to_string(b_rows) + "," + std::to_string(b_cols) +
                ") is invalid"
            );
        }
        return;
    }

    throw std::runtime_error("Unknown op: " + op_name);
}

// ---- Parent function builder ----

NodePtr Tensor::make_parent_fn(const TensorPtr& t) {
    if (t->is_leaf()) {
        if (!t->accumulate_node) {
            t->accumulate_node = std::make_shared<AccumulateGrad>(t);
        }
        // std::shared_ptr<AccumulateGrad> -> NodePtr
        return t->accumulate_node;
    } else if (t->requires_grad && t->grad_fn) {
        return t->grad_fn;
    } else {
        return nullptr;
    }
}

// ---- Backward graph traversal ----

namespace {
void build_topo(const NodePtr& node,
                std::unordered_set<Node*>& visited,
                std::vector<NodePtr>& topo) {
    if (!node) return;

    Node* raw = node.get();
    if (visited.count(raw)) {
        return;
    }
    visited.insert(raw);

    for (auto& wparent : node->parents) {
        if (auto parent = wparent.lock()) {
            build_topo(parent, visited, topo);
        }
    }

    topo.push_back(node);
}
}

void Tensor::backward() {
    if (!requires_grad) {
        return;
    }

    // Seed gradient at this tensor: d(loss)/d(this)
    Matrix grad_out = Matrix::Ones(data.rows(), data.cols());

    if (!grad_fn) {
        if (!grad_initialized) {
            grad = grad_out;
            grad_initialized = true;
        } else {
            grad += grad_out;
        }
        return;
    }

    // Build topological order of nodes 
    std::unordered_set<Node*> visited;
    std::vector<NodePtr> topo;
    build_topo(grad_fn, visited, topo);

    std::unordered_map<Node*, Matrix> node_grads;
    node_grads[grad_fn.get()] = grad_out;

    // Process nodes from root toward leaves
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        NodePtr node = *it;
        Node* raw = node.get();

        auto itg = node_grads.find(raw);
        if (itg == node_grads.end()) {
            continue; // no gradient flowed to this node
        }

        const Matrix& g = itg->second;

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
                itp->second += *opt_pg;
            }
        }
    }
}

}