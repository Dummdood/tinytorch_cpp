#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.hpp"
#include "module.hpp"
#include "linear.hpp"
#include "optimizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tinytorch_cpp, m) {
    m.doc() = "TinyTorch-CPP Python bindings";

    // ----------------------------
    // Tensor
    // ----------------------------
    py::class_<Tensor>(m, "Tensor")
        // Constructors
        .def(py::init<int, int, bool>(),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("requires_grad") = false)

        // Basic properties
        .def_property_readonly("rows", &Tensor::rows)
        .def_property_readonly("cols", &Tensor::cols)
        .def_property_readonly("shape", [](const Tensor& t) {
            return std::make_pair(t.rows(), t.cols());
        })
        .def_property("requires_grad",
                      &Tensor::requires_grad,
                      &Tensor::set_requires_grad)

        // Grad access: grad or None
        .def_property_readonly("grad", [](Tensor& t) -> py::object {
            if (!t.has_grad()) {
                return py::none();
            }
            // Expose a reference so Python sees live updates
            return py::cast(std::ref(t.grad()));
        })

        // Methods
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", &Tensor::backward)
        .def("item", &Tensor::item)

        // Indexing: t[i, j]
        .def("__getitem__", [](const Tensor& t, std::pair<int, int> idx) {
            return t(idx.first, idx.second);
        })
        .def("__setitem__", [](Tensor& t, std::pair<int, int> idx, float value) {
            t(idx.first, idx.second) = value;
        })

        // Optional: simple repr
        .def("__repr__", [](const Tensor& t) {
            return "<Tensor shape=(" +
                   std::to_string(t.rows()) + "," +
                   std::to_string(t.cols()) + ")>";
        });

    // mse_loss as a free function
    m.def("mse_loss",
          &Tensor::mse_loss,
          py::arg("pred"),
          py::arg("target"),
          "Mean squared error loss returning a scalar Tensor");

    // ----------------------------
    // Module base class
    // ----------------------------
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("parameters", &Module::parameters)
        .def("zero_grad", &Module::zero_grad);

    // ----------------------------
    // Linear
    // ----------------------------
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>(),
             py::arg("in_features"),
             py::arg("out_features"))
        .def("forward", &Linear::forward);

    // ----------------------------
    // Sequential
    // ----------------------------
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<>())
        .def("add", &Sequential::add)
        .def("forward", &Sequential::forward)
        .def("parameters", &Sequential::parameters)
        .def("zero_grad", &Sequential::zero_grad);

    // ----------------------------
    // ReLU module
    // ----------------------------
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("parameters", &ReLU::parameters);

    // ----------------------------
    // Sigmoid module
    // ----------------------------
    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward)
        .def("parameters", &Sigmoid::parameters);

    // ----------------------------
    // SGD optimizer
    // ----------------------------
    py::class_<SGD>(m, "SGD")
        .def(py::init<float>(), py::arg("lr"))
        .def("step", [](SGD& opt, Module& m) {
            opt.step(m);
        }, py::arg("module"));
}