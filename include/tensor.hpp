#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include "node.hpp"
#include <utility>

class Tensor {
    // Data members
    int rows_;
    int cols_;
    std::vector<float> vals_;

    bool requires_grad_ = false;
    std::unique_ptr<Tensor> grad_;
    std::shared_ptr<Node> grad_fn_ = nullptr;

    // Getters
    int getIndex(int row, int col) const {
        if (row >= 0 && row < rows_ && col >= 0 && col < cols_) {
            return row * cols_ + col;
        }
        throw std::invalid_argument(
            "Tensor: index (" + std::to_string(row) + ", " + std::to_string(col) +
            ") is out of bounds for shape (" + std::to_string(rows_) +
            ", " + std::to_string(cols_) + ")."
        );
    }

    // Error Checks
    static void initCheck(int rows, int cols) {
        if (rows < 1 || cols < 1) {
            throw std::invalid_argument("Tensor: shape must be strictly positive in both dimensions."); 
        }
    }
    static void tMatch(const Tensor& t1, const Tensor& t2) {
        if (t1.rows_ != t2.rows_ || t1.cols_ != t2.cols_) {
            throw std::invalid_argument("Tensor: shapes must match.");
        }
    }
    static void matmulMatch(const Tensor& t1, const Tensor& t2) {
        if (t1.cols_ != t2.rows_) {
            throw std::invalid_argument("Tensor: matmul dimension mismatch (t1.cols != t2.rows).");
        }
    }

public:
    // Constructors
    explicit Tensor(int rows, int cols, bool flag = false) 
        : rows_(rows), cols_(cols), requires_grad_(flag) {
        initCheck(rows, cols);
        vals_.resize(rows * cols);
    }

    // Getters
    int rows() const {
        return rows_;
    }
    int cols() const {
        return cols_;
    }
    int numel() const {
        return rows_ * cols_;
    }
    bool requires_grad() const {
        return requires_grad_;
    }
    bool has_grad() const {
        return static_cast<bool>(grad_);
    }
    const Tensor& grad() const {
        if (!grad_) {
            throw std::runtime_error("Tensor has no gradient yet");
        }
        return *grad_;
    }
    Tensor& grad() {
        if (!grad_) {
            grad_ = std::make_unique<Tensor>(rows_, cols_);
            // // Initialize to 0
            for (int i = 0; i < rows_; ++i)
                for (int j = 0; j < cols_; ++j)
                    grad_->operator()(i, j) = 0.0f;
        }
        return *grad_;
    }
    bool is_leaf() const {
        return grad_fn_ == nullptr && requires_grad_;
    }
    std::shared_ptr<Node> grad_fn() const {
        return grad_fn_;
    }

    // Setters
    void set_requires_grad(bool grad) {
        requires_grad_ = grad;
    }
    void zero_grad() {
        if (grad_) {
            for (int i = 0; i < rows_; ++i)
                for (int j = 0; j < cols_; ++j)
                    grad_->operator()(i, j) = 0.0f;
        }
    }
    void set_grad_fn(std::shared_ptr<Node> fn) {
        grad_fn_ = std::move(fn);
    }

    // Factories
    static Tensor full(int rows, int cols, float value) {
        Tensor t(rows, cols);
        std::fill(t.vals_.begin(), t.vals_.end(), value);
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
        for (float& val : t.vals_) {
            val = dist(gen);
        }
        return t;
    }
    static Tensor normal(int rows, int cols, float mean, float stdev) {
        Tensor t(rows, cols);
        static std::mt19937 gen(12345);
        std::normal_distribution<float> dist(mean, stdev);
        for (float& val : t.vals_) {
            val = dist(gen);
        }
        return t;
    }

    // Operations Overloads
    float& operator()(int row, int col) {
        return vals_[getIndex(row, col)];
    }
    const float& operator()(int row, int col) const {
        return vals_[getIndex(row, col)];
    }
    static Tensor add(const Tensor& t1, const Tensor& t2) {
        tMatch(t1, t2);
        int R = t1.rows_, C = t1.cols_;

        Tensor sum(R, C);
        std::transform(
            t1.vals_.begin(), t1.vals_.end(),
            t2.vals_.begin(),
            sum.vals_.begin(),
            std::plus<float>()
        );

        bool grad = t1.requires_grad_ || t2.requires_grad_;
        if (!grad) {
            return sum;
        }

        auto node = std::make_shared<Node>();
        node->parents = {
            const_cast<Tensor*>(&t1),
            const_cast<Tensor*>(&t2)
        };

        node->backward =
            [p1 = const_cast<Tensor*>(&t1),
            p2 = const_cast<Tensor*>(&t2),
            R, C](const Tensor& grad_out)
        {
            if (p1->requires_grad_) {
                Tensor& g1 = p1->grad();
                for (int i = 0; i < R; ++i)
                    for (int j = 0; j < C; ++j)
                        g1(i, j) += grad_out(i, j);
            }

            if (p2->requires_grad_) {
                Tensor& g2 = p2->grad();
                for (int i = 0; i < R; ++i)
                    for (int j = 0; j < C; ++j)
                        g2(i, j) += grad_out(i, j);
            }
        };

        sum.requires_grad_ = true;
        sum.set_grad_fn(node);

        return sum;
    }
    static Tensor mul(const Tensor& t1, const Tensor& t2) {
        tMatch(t1, t2);
        Tensor prod(t1.rows_, t1.cols_);
        std::transform(t1.vals_.begin(), t1.vals_.end(),
            t2.vals_.begin(),
            prod.vals_.begin(),
            std::multiplies<float>()); 
        return prod;
    }
    static Tensor matmul(const Tensor& t1, const Tensor& t2) {
        matmulMatch(t1, t2);
        int M = t1.rows_, K = t1.cols_, N = t2.cols_;

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

        bool requires = t1.requires_grad_ || t2.requires_grad_;
        if (!requires) {
            return out;
        }

        auto node = std::make_shared<Node>();
        node->parents = {
            const_cast<Tensor*>(&t1),
            const_cast<Tensor*>(&t2)
        };

        node->backward =
            [pA = const_cast<Tensor*>(&t1),
            pB = const_cast<Tensor*>(&t2),
            M, K, N](const Tensor& grad_out)
        {
            // grad_out shape: (M × N)

            // dA = grad_out.matmul(Bᵀ)
            if (pA->requires_grad_) {
                Tensor Bt = pB->transpose();             // (N × K)
                Tensor dA = matmul(grad_out, Bt);        // (M × K)

                Tensor& gA = pA->grad();                 // accumulate
                for (int i = 0; i < M; ++i)
                    for (int j = 0; j < K; ++j)
                        gA(i, j) += dA(i, j);
            }

            // dB = Aᵀ.matmul(grad_out)
            if (pB->requires_grad_) {
                Tensor At = pA->transpose();             // (K × M)
                Tensor dB = matmul(At, grad_out);        // (K × N)

                Tensor& gB = pB->grad();                 // accumulate
                for (int i = 0; i < K; ++i)
                    for (int j = 0; j < N; ++j)
                        gB(i, j) += dB(i, j);
            }
        };

        out.requires_grad_ = true;
        out.set_grad_fn(node);

        return out;
    }
    Tensor transpose() const {
        Tensor out(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                out(j, i) = (*this)(i, j);
            }
        }
        return out;
    }

    // Activation Functions
    static Tensor relu(const Tensor& t) {
        Tensor out(t.rows_, t.cols_);
        std::transform(
            t.vals_.begin(),
            t.vals_.end(),
            out.vals_.begin(),
            [](float x) { return x > 0.0f ? x : 0.0f; }
        );
        return out;
    }
    static Tensor sigmoid(const Tensor& t) {
        Tensor out(t.rows_, t.cols_);
        std::transform(
            t.vals_.begin(),
            t.vals_.end(),
            out.vals_.begin(),
            [](float x) { return 1.0f / (1.0f + std::exp(-x)); }
        );
        return out;
    }
    static Tensor tanh(const Tensor& t) {
        Tensor out(t.rows_, t.cols_);
        std::transform(
            t.vals_.begin(),
            t.vals_.end(),
            out.vals_.begin(),
            [](float x) { return std::tanh(x); }
        );
        return out;
    }
    static Tensor softmax(const Tensor& t) {
        float max_logit = *std::max_element(t.vals_.begin(), t.vals_.end());

        Tensor expT(t.rows_, t.cols_);
        std::transform(
            t.vals_.begin(),
            t.vals_.end(),
            expT.vals_.begin(),
            [max_logit](float x) { return std::exp(x - max_logit); }
        );

        float sum_exp = std::accumulate(expT.vals_.begin(), expT.vals_.end(), 0.0f);
        std::transform(
            expT.vals_.begin(),
            expT.vals_.end(),
            expT.vals_.begin(),
            [sum_exp](float x) { return x / sum_exp ; }
        );
        return expT;
    }
};

// Operation Overloads
inline Tensor operator+(const Tensor& t1, const Tensor& t2) {
    return Tensor::add(t1, t2);
}
inline Tensor operator*(const Tensor& t1, const Tensor& t2) {
    return Tensor::mul(t1, t2);
}