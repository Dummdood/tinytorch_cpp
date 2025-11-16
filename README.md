# TinyTorch-CPP

TinyTorch-CPP is a small deep learning framework written in C++17 with a Python frontend via `pybind11`.  
The goals are:

- Learn how frameworks like PyTorch work under the hood
- Implement a minimal autograd engine
- Build simple neural networks from both C++ and Python

---

## Features

- **Tensor**
  - 2D tensors stored as `std::vector<float>`
  - Shape: `(rows, cols)`
  - Element access: `t(row, col)`
  - Factory helpers: `zeros`, `ones`, `randn`, `normal`, `full`
  - Ops with autograd: `add`, `mul`, `matmul`, `relu`, `sigmoid`, `sum`, `mse_loss`
  - Autograd:
    - `requires_grad` flag
    - `.grad()` storage (lazy allocation)
    - `.backward()` for scalar tensors (1×1)

- **Autograd graph**
  - `Node` struct with:
    - `std::vector<Tensor*> parents`
    - `std::function<void(const Tensor&)> backward`
  - Each differentiable op creates a `Node` and attaches it to its output tensor as `grad_fn_`

- **Modules / Layers**
  - `Module` base class
    - `virtual Tensor forward(const Tensor& x)`
    - `virtual std::vector<Tensor*> parameters()`
    - `void zero_grad()`
  - `Linear`: fully-connected layer `y = xW + b`
  - `Sequential`: container that applies submodules in order
  - `ReLU` and `Sigmoid` as `Module`s (no parameters)

- **Optimizer**
  - `SGD` with:
    - `SGD(float lr)`
    - `step(const std::vector<Tensor*>& params)`
    - `step(Module& m)` convenience

- **Python bindings**
  - Built with `pybind11` as a submodule under `extern/pybind11`
  - Python extension module: `tinytorch_cpp`
  - Exposes (at minimum):
    - `Tensor`
    - `mse_loss`
    - (optionally) `Linear`, `Sequential`, `ReLU`, `Sigmoid`, `SGD`

---

## Directory Layout

```text
TINYTORCH-CPP/
  include/
    tensor.hpp
    node.hpp
    module.hpp
    linear.hpp
    optimizer.hpp
    ... (other headers)
  src/
    main.cpp          # C++ demo / playground
    bindings.cpp      # pybind11 bindings for Python
  extern/
    pybind11/         # pybind11 submodule (git submodule)
  CMakeLists.txt
  README.md