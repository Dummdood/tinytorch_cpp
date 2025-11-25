#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>   // <=== ADD THIS

#include "tensor.hpp"
#include "loss.hpp"
#include "nn.hpp"
#include "optim.hpp"

namespace py = pybind11;
using namespace autodiff;

PYBIND11_MODULE(tinytorch_cpp, m) {
    m.doc() = "TinyTorch C++ backend exposed to Python";

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        // ---- Constructor from numpy ----
        .def(py::init([](Eigen::MatrixXd mat, bool requires_grad) {
            return std::make_shared<Tensor>(mat, requires_grad);
        }))

        // ---- data / grad as numpy arrays ----
        .def_property(
            "data",
            [](Tensor& t) { return t.data; },
            [](Tensor& t, const Eigen::MatrixXd& m) { t.data = m; }
        )
        .def_property(
            "grad",
            [](Tensor& t) { return t.grad; },
            [](Tensor& t, const Eigen::MatrixXd& m) {
                t.grad = m;
                t.grad_initialized = true;
            }
        )

        .def("zero_grad", [](Tensor& t) {
            t.grad.setZero();
            t.grad_initialized = false;
        })
        .def("backward", &Tensor::backward)

        // ---- Operator overloads ----
        .def("__add__", [](const TensorPtr& a, const TensorPtr& b) {
            return Tensor::add(a, b);
        })
        .def("__sub__", [](const TensorPtr& a, const TensorPtr& b) {
            return Tensor::sub(a, b);
        })
        .def("__mul__", [](const TensorPtr& a, const TensorPtr& b) {
            return Tensor::mul(a, b);
        })
        .def("__truediv__", [](const TensorPtr& a, const TensorPtr& b) {
            return Tensor::div(a, b);
        })
        .def("__matmul__", [](const TensorPtr& a, const TensorPtr& b) {
            return Tensor::matmul(a, b);
        })
        ;

    // Loss
    m.def("mse_loss", &mse_loss);

    // Linear
    py::class_<Linear>(m, "Linear")
        .def(py::init<int,int,bool>())
        .def("__call__", &Linear::operator())
        .def("parameters", &Linear::parameters)
        ;

    // MLP
    py::class_<MLP>(m, "MLP")
        .def(py::init<int, std::vector<int>, int, std::string, std::string>())
        .def("__call__", &MLP::operator())
        .def("parameters", &MLP::parameters)
        ;

    // SGD
    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<TensorPtr>, double>())
        .def("zero_grad", &SGD::zero_grad)
        .def("step", &SGD::step)
        ;
}