#pragma once
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>


// Forward declarations
class Tensor;         // forward
struct TensorImpl;    // forward

struct Node {
    // Each parent is identified by the shared impl it uses
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void(const Tensor&)> backward;
};

// Shared implementation object
struct TensorImpl {
    int rows;
    int cols;
    std::vector<float> vals;

    bool requires_grad;
    std::unique_ptr<Tensor> grad;       // gradient storage
    std::shared_ptr<Node> grad_fn;      // autograd node

    TensorImpl(int r, int c, bool req)
        : rows(r),
          cols(c),
          vals(r * c),
          requires_grad(req),
          grad(nullptr),
          grad_fn(nullptr)
    {}

    ~TensorImpl(); // defined after Tensor
};

class Tensor {
    // Single data member: handle to shared impl
    std::shared_ptr<TensorImpl> impl_;

    // Allow autograd to re-wrap an existing impl
    explicit Tensor(std::shared_ptr<TensorImpl> impl)
        : impl_(std::move(impl)) {}

    // Getters
    int getIndex(int row, int col) const {
        if (row >= 0 && row < impl_->rows && col >= 0 && col < impl_->cols) {
            return row * impl_->cols + col;
        }
        throw std::invalid_argument(
            "Tensor: index (" + std::to_string(row) + ", " + std::to_string(col) +
            ") is out of bounds for shape (" + std::to_string(impl_->rows) +
            ", " + std::to_string(impl_->cols) + ")."
        );
    }

    // Error Checks
    static void initCheck(int rows, int cols) {
        if (rows < 1 || cols < 1) {
            throw std::invalid_argument("Tensor: shape must be strictly positive in both dimensions."); 
        }
    }
    static void tMatch(const Tensor& t1, const Tensor& t2) {
        if (t1.rows() != t2.rows() || t1.cols() != t2.cols()) {
            throw std::invalid_argument("Tensor: shapes must match.");
        }
    }
    static void matmulMatch(const Tensor& t1, const Tensor& t2) {
        if (t1.cols() != t2.rows()) {
            throw std::invalid_argument("Tensor: matmul dimension mismatch (t1.cols != t2.rows).");
        }
    }
    void isScalar() const {
        if (impl_->rows != 1 || impl_->cols != 1) {
            throw std::runtime_error("item() only valid for scalar tensors");
        }
    }

public:
    // Constructors
    explicit Tensor(int rows, int cols, bool requires_grad = false)
        : impl_(std::make_shared<TensorImpl>(rows, cols, requires_grad)) {
        initCheck(rows, cols);
    }

    // Defaulted copy/move: share impl_
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;


    // Basic Getters
    int rows() const { return impl_->rows; }
    int cols() const { return impl_->cols; }
    int numel() const { return impl_->rows * impl_->cols; }
    float item() const { isScalar(); return (*this)(0, 0); }

    // Grad Getters
    bool requires_grad() const { return impl_->requires_grad; }
    bool has_grad() const { return static_cast<bool>(impl_->grad); }
    const Tensor& grad() const {
        if (!impl_->grad) {
            throw std::runtime_error("Tensor has no gradient yet");
        }
        return *impl_->grad;
    }
    Tensor& grad() {
        if (!impl_->grad) {
            impl_->grad = std::make_unique<Tensor>(rows(), cols(), false);
            Tensor& g = *impl_->grad;
            for (int i = 0; i < rows(); ++i)
                for (int j = 0; j < cols(); ++j)
                    g(i, j) = 0.0f;
        }
        return *impl_->grad;
    }
    std::shared_ptr<Node> grad_fn() const { return impl_->grad_fn; }

    // Setters
    void set_requires_grad(bool grad) { impl_->requires_grad = grad; }
    void set_grad_fn(const std::shared_ptr<Node>& fn) { impl_->grad_fn = fn; }
    void zero_grad() {
        if (impl_->grad) {
            Tensor& g = *impl_->grad;
            for (int i = 0; i < rows(); ++i)
                for (int j = 0; j < cols(); ++j)
                    g(i, j) = 0.0f;
        }
    }

    // Element access
    float& operator()(int row, int col) {
        return impl_->vals[getIndex(row, col)];
    }
    const float& operator()(int row, int col) const {
        return impl_->vals[getIndex(row, col)];
    }

    // Factories
    static Tensor full(int rows, int cols, float value) {
        Tensor t(rows, cols);
        std::fill(t.impl_->vals.begin(), t.impl_->vals.end(), value);
        return t;
    }
    static Tensor zeros(int rows, int cols) {
        return full(rows, cols, 0.0f);
    }
    static Tensor ones(int rows, int cols) {
        return full(rows, cols, 1.0f);
    }
    static Tensor randn(int rows, int cols) {
        Tensor t(rows, cols);
        static std::mt19937 gen(12345);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float& val : t.impl_->vals) {
            val = dist(gen);
        }
        return t;
    }
    static Tensor normal(int rows, int cols, float mean, float stdev) {
        Tensor t(rows, cols);
        static std::mt19937 gen(12345);
        std::normal_distribution<float> dist(mean, stdev);
        for (float& val : t.impl_->vals) {
            val = dist(gen);
        }
        return t;
    }

    // Operations Overloads
    static Tensor add(const Tensor& t1, const Tensor& t2) {
        tMatch(t1, t2);
        int R = t1.rows();
        int C = t1.cols();

        // Forward
        Tensor sum(R, C);
        std::transform(
            t1.impl_->vals.begin(), t1.impl_->vals.end(),
            t2.impl_->vals.begin(),
            sum.impl_->vals.begin(),
            std::plus<float>()
        );

        bool grad = t1.requires_grad() || t2.requires_grad();
        if (!grad) {
            return sum;
        }

        auto node = std::make_shared<Node>();
        node->parents = { t1.impl_, t2.impl_ };

        node->backward =
            [t1_impl = t1.impl_, t2_impl = t2.impl_, R, C](const Tensor& grad_out)
        {
            Tensor t1(t1_impl);
            Tensor t2(t2_impl);

            // dL/dt1 += grad_out
            if (t1.requires_grad()) {
                Tensor& g1 = t1.grad();
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        g1(i, j) += grad_out(i, j);
                    }
                }
            }

            // dL/dt2 += grad_out
            if (t2.requires_grad()) {
                Tensor& g2 = t2.grad();
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        g2(i, j) += grad_out(i, j);
                    }
                }
            }
        };

        Tensor sum_autograd(R, C, true);
        sum_autograd.impl_->vals = sum.impl_->vals;
        sum_autograd.set_grad_fn(node);
        return sum_autograd;
    }
    static Tensor sum(const Tensor& t) {
        // Forward: scalar (1×1)
        Tensor out(1, 1, t.requires_grad());
        float total = std::accumulate(
            t.impl_->vals.begin(),
            t.impl_->vals.end(),
            0.0f
        );
        out(0, 0) = total;

        if (!t.requires_grad()) {
            return out;
        }

        auto node = std::make_shared<Node>();
        node->parents = { t.impl_ };

        int R = t.rows();
        int C = t.cols();

        node->backward =
            [t_impl = t.impl_, R, C](const Tensor& grad_out)
        {
            Tensor t(t_impl);
            if (!t.requires_grad()) return;

            float g_scalar = grad_out(0, 0);  // dL/d(sum) is scalar

            Tensor& g = t.grad();
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    g(i, j) += g_scalar;  // same gradient for every element
                }
            }
        };

        out.set_requires_grad(true);
        out.set_grad_fn(node);
        return out;
    }
    static Tensor mul(const Tensor& t1, const Tensor& t2) {
        tMatch(t1, t2);
        int R = t1.rows();
        int C = t1.cols();

        // Forward
        Tensor prod(R, C);
        std::transform(
            t1.impl_->vals.begin(), t1.impl_->vals.end(),
            t2.impl_->vals.begin(),
            prod.impl_->vals.begin(),
            std::multiplies<float>()
        );

        bool requires = t1.requires_grad() || t2.requires_grad();
        if (!requires) {
            return prod;
        }

        auto node = std::make_shared<Node>();
        // Parents identified by their impl handles
        node->parents = { t1.impl_, t2.impl_ };

        node->backward =
            [t1_impl = t1.impl_, t2_impl = t2.impl_, R, C](const Tensor& grad_out)
        {
            Tensor t1(t1_impl);
            Tensor t2(t2_impl);

            // dL/dt1 += grad_out * t2
            if (t1.requires_grad()) {
                Tensor& g1 = t1.grad();
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        g1(i, j) += grad_out(i, j) * t2(i, j);
                    }
                }
            }

            // dL/dt2 += grad_out * t1
            if (t2.requires_grad()) {
                Tensor& g2 = t2.grad();
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        g2(i, j) += grad_out(i, j) * t1(i, j);
                    }
                }
            }
        };

        prod.set_requires_grad(true);
        prod.set_grad_fn(node);
        return prod;
    }
    static Tensor matmul(const Tensor& t1, const Tensor& t2) {
        matmulMatch(t1, t2);
        int M = t1.rows();
        int K = t1.cols();
        int N = t2.cols();

        // Forward: out = t1 @ t2
        Tensor out = zeros(M, N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += t1(i, k) * t2(k, j);
                }
                out(i, j) = sum;
            }
        }

        bool requires = t1.requires_grad() || t2.requires_grad();
        if (!requires) {
            return out;
        }

        auto node = std::make_shared<Node>();
        // Store parents by their shared impl handles
        node->parents = { t1.impl_, t2.impl_ };

        node->backward =
            [A_impl = t1.impl_, B_impl = t2.impl_, M, K, N](const Tensor& grad_out)
        {
            // Wrap impls into Tensor handles
            Tensor A(A_impl);
            Tensor B(B_impl);

            // grad_out shape: (M × N)

            // dA = grad_out @ Bᵀ
            if (A.requires_grad()) {
                Tensor& gA = A.grad();  // shape (M × K)
                for (int i = 0; i < M; ++i) {
                    for (int k = 0; k < K; ++k) {
                        float acc = 0.0f;
                        for (int j = 0; j < N; ++j) {
                            acc += grad_out(i, j) * B(k, j); // Bᵀ(j,k) = B(k,j)
                        }
                        gA(i, k) += acc;
                    }
                }
            }

            // dB = Aᵀ @ grad_out
            if (B.requires_grad()) {
                Tensor& gB = B.grad();  // shape (K × N)
                for (int k = 0; k < K; ++k) {
                    for (int j = 0; j < N; ++j) {
                        float acc = 0.0f;
                        for (int i = 0; i < M; ++i) {
                            acc += A(i, k) * grad_out(i, j); // Aᵀ(k,i) = A(i,k)
                        }
                        gB(k, j) += acc;
                    }
                }
            }
        };

        out.set_requires_grad(true);
        out.set_grad_fn(node);
        return out;
    }
    Tensor transpose() const {
        int rows = impl_->rows, cols = impl_->cols;
        Tensor out(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out(j, i) = (*this)(i, j);
            }
        }
        return out;
    }
    void backward() {
        isScalar();

        // Seed gradient of the root
        Tensor& root_grad = grad();
        root_grad(0, 0) = 1.0f;

        // Worklist over TensorImpl handles
        std::vector<std::shared_ptr<TensorImpl>> stack;
        stack.push_back(impl_);

        while (!stack.empty()) {
            auto cur_impl = stack.back();
            stack.pop_back();

            if (!cur_impl) continue;

            Tensor t(cur_impl);               // wrap impl in a Tensor handle
            auto node = cur_impl->grad_fn;    // autograd node
            if (!node) continue;

            const Tensor& grad_out = t.grad();  // dL/dt

            // Call node-specific backward
            node->backward(grad_out);

            // Traverse parents by impl handle
            for (auto& parent_impl : node->parents) {
                if (!parent_impl) continue;
                Tensor parent(parent_impl);
                if (parent.requires_grad()) {
                    stack.push_back(parent_impl);
                }
            }
        }
    }

    // Activation Functions
    static Tensor relu(const Tensor& t) {
        int R = t.rows(), C = t.cols();

        // Forward
        Tensor out(R, C);
        std::transform(
            t.impl_->vals.begin(),
            t.impl_->vals.end(),
            out.impl_->vals.begin(),
            [](float x) { return x > 0.0f ? x : 0.0f; }
        );

        // Autograd?
        if (!t.requires_grad()) { 
            return out; 
        }

        auto node = std::make_shared<Node>();
        node->parents = { t.impl_ };

        node->backward =
            [t_impl = t.impl_, R, C](const Tensor& grad_out)
        {
            Tensor t(t_impl);
            if (!t.requires_grad()) return;

            Tensor& g = t.grad();
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    float x = t(i, j);
                    float mask = (x > 0.0f) ? 1.0f : 0.0f;
                    g(i, j) += grad_out(i, j) * mask;
                }
            }
        };

        out.set_requires_grad(true);
        out.set_grad_fn(node);
        return out;
    }
    static Tensor sigmoid(const Tensor& t) {
        int R = t.rows(), C = t.cols();

        // Forward
        Tensor out(R, C);
        std::transform(
            t.impl_->vals.begin(),
            t.impl_->vals.end(),
            out.impl_->vals.begin(),
            [](float x) { return 1.0f / (1.0f + std::exp(-x)); }
        );

        // Autograd?
        if (!t.requires_grad()) {
            return out;
        }

        auto node = std::make_shared<Node>();
        node->parents = { t.impl_ };

        node->backward =
            [t_impl = t.impl_, R, C](const Tensor& grad_out)
        {
            Tensor t(t_impl);
            if (!t.requires_grad()) return;

            Tensor& g = t.grad();
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    float x = t(i, j);
                    float s = 1.0f / (1.0f + std::exp(-x));
                    g(i, j) += grad_out(i, j) * s * (1.0f - s);
                }
            }
        };

        out.set_requires_grad(true);
        out.set_grad_fn(node);
        return out;
    }
    
    // Loss Functions
    static Tensor mse_loss(const Tensor& pred, const Tensor& target) {
        // Shape check
        tMatch(pred, target);
        int R = pred.rows(), C = pred.cols();
        int N = R * C;

        // ---- Forward: scalar loss ----
        Tensor out(1, 1);
        float total = 0.0f;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                float diff = pred(i, j) - target(i, j);
                total += diff * diff;
            }
        }
        out(0, 0) = total / static_cast<float>(N);

        // ---- Autograd? ----
        bool requires = pred.requires_grad() || target.requires_grad();
        if (!requires) { 
            return out; 
        }

        auto node = std::make_shared<Node>();
        node->parents = { pred.impl_, target.impl_ };

        node->backward =
            [p_impl = pred.impl_, t_impl = target.impl_, R, C, N](const Tensor& grad_out)
        {
            Tensor p(p_impl);
            Tensor t(t_impl);

            float g = grad_out(0, 0);  // upstream grad (scalar)
            float scale = (2.0f * g) / static_cast<float>(N);

            // dL/dpred
            if (p.requires_grad()) {
                Tensor& gp = p.grad();
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        float diff = p(i, j) - t(i, j);
                        gp(i, j) += scale * diff;
                    }
                }
            }

            // dL/dtarget
            if (t.requires_grad()) {
                Tensor& gt = t.grad();
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        float diff = p(i, j) - t(i, j);
                        gt(i, j) += -scale * diff;
                    }
                }
            }
        };

        out.set_requires_grad(true);
        out.set_grad_fn(node);
        return out;
    }
};
inline TensorImpl::~TensorImpl() = default;

// Operation Overloads
inline Tensor operator+(const Tensor& t1, const Tensor& t2) {
    return Tensor::add(t1, t2);
}
inline Tensor operator*(const Tensor& t1, const Tensor& t2) {
    return Tensor::mul(t1, t2);
}