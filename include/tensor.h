#include <vector>
#include <stdexcept>
#include <random>

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

    static void tMatch(const Tensor& t1, const Tensor& t2) {
        if (t1.rows_ != t2.rows_ || t1.cols_ != t2.cols_) {
            throw std::invalid_argument("Tensor: shapes must match.");
        }
    }

public:
    // Constructors
    explicit Tensor(int rows, int cols) : rows_(rows), cols_(cols) {
        if (rows < 1 || cols < 1) {
            throw std::invalid_argument("Tensor: shape must be strictly positive in both dimensions."); 
        }
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

    // Operators
    float& operator()(int row, int col) {
        return vals_[getIndex(row, col)];
    }
    const float& operator()(int row, int col) const {
        return vals_[getIndex(row, col)];
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

    // Operations
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
};

// Operation Overloads
inline Tensor operator+(const Tensor& t1, const Tensor& t2) {
    return Tensor::add(t1, t2);
}
inline Tensor operator*(const Tensor& t1, const Tensor& t2) {
    return Tensor::mul(t1, t2);
}