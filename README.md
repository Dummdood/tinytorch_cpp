# TinyTorch-CPP

A tiny, educational autograd engine and neural-net playground written in modern C++17, with:
- a minimal Tensor + dynamic computation graph (`grad_fn` nodes)
- backward passes for common ops (add, mul, matmul, relu, sigmoid, exp, log, sum, mean, pow, …)
- a small `nn` layer stack (`Linear`, `MLP`)
- a simple `SGD` optimizer
- Python bindings via **pybind11** so you can train from Python while executing the core in C++

This repo also includes a pure-Python “reference” TinyTorch and benchmark scripts to compare:
- C++ executable
- C++ core called from Python (pybind module)
- pure Python TinyTorch
- PyTorch baseline

## Repository layout
tinytorch-cpp/
│
├── CMakeLists.txt
├── README.md
├── .gitmodules
├── .gitignore
│
├── extern/
│   └── pybind11/                # pybind11 submodule
│
├── include/
│   ├── tensor.hpp
│   ├── backward_nodes.hpp
│   ├── loss.hpp
│   ├── optim.hpp
│   └── nn.hpp
│
├── src/
│   ├── tensor.cpp
│   ├── backward_nodes.cpp
│   ├── loss.cpp
│   ├── optim.cpp                # optional (may be empty or omitted)
│   ├── nn.cpp
│   ├── bindings.cpp             # pybind11 module implementation
│   └── tinytorch_cpp_bench.cpp  # C++ benchmark executable
│
├── build/                       # created by CMake
│   └── tinytorch_cpp*.so        # python extension after build
│
└── tinytorch-python/
    └── tinytorch_py/
        ├── tensor.py
        ├── losses.py
        ├── optim/
        │   ├── sgd.py
        │   └── ...
        ├── nn/
        │   ├── mlp.py
        │   └── ...
        ├── backward_nodes.py
        ├── engine.py
        ├── tinytorch_cpp_bench.py    # Python → C++ benchmark
        ├── tinytorch_py_bench.py     # pure Python benchmark
        └── pytorch_bench.py          # PyTorch baseline


## What’s implemented (C++ core)

- **Autograd graph:** `Tensor` holds `grad_fn` (`NodePtr`) and gradients are accumulated in leaf nodes via `AccumulateGrad`.
- **Backward nodes:** per-op `Node::apply(grad_output)` returning gradients to parents.
- **SGD:** `zero_grad()` + `step()`.
- **NN blocks:** `Linear` and `MLP` with trainable parameters exposed via `.parameters()`.

Limitations (by design, for learning):
- no GPU, no broadcasting (unless you added it), no batching abstractions beyond what you manually code
- not optimized for performance yet (allocations, per-sample loops, Eigen expression choices, etc.)

## Build prerequisites

- CMake (>= 3.14)
- A C++17 compiler (Clang on macOS works)
- Eigen (installed and available on your include path)
- Python 3.x (if building bindings)
- pybind11 is included as a submodule in `extern/pybind11`

## Build (C++ bench + Python module)

From the repo root:

```bash
git submodule update --init --recursive

cmake -S . -B build
cmake --build build -j
