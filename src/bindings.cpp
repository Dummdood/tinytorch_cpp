#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tinytorch_cpp, m) {
    m.doc() = "TinyTorch-CPP Python bindings";

    // Bind Tensor
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

    // Expose mse_loss as a module-level function
    m.def("mse_loss",
          &Tensor::mse_loss,
          py::arg("pred"),
          py::arg("target"),
          "Mean squared error loss returning a scalar Tensor");
}