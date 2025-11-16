#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>

class Tensor {
    int rows_;
    int cols_;
    std::vector<float> vals_;
    
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

    // Error checking
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
    explicit Tensor(int rows, int cols) : rows_(rows), cols_(cols) {
        initCheck(rows, cols);
        vals_.resize(rows * cols);
    }

    // Accessors
    int rows() const {
        return rows_;
    }
    int cols() const {
        return cols_;
    }
    int numel() const {
        return rows_ * cols_;
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
        Tensor sum(t1.rows_, t1.cols_);
        std::transform(t1.vals_.begin(), t1.vals_.end(),
            t2.vals_.begin(),
            sum.vals_.begin(),
            std::plus<float>()); 
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
        Tensor out = zeros(t1.rows_, t2.cols_);
        for (int i = 0; i < t1.rows_; i++) {
            for (int j = 0; j < t2.cols_; j++) {
                for (int k = 0; k < t1.cols_; k++) {
                    out(i, j) += t1(i, k) * t2(k, j);   // direct accumulation
                }
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