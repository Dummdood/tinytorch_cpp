# TinyTorch-CPP

TinyTorch-CPP is a lightweight, dependency-minimal deep learning engine implemented in modern C++17, 
designed to run MLPs (multilayer perceptrons) where full-scale frameworks like PyTorch introduce unnecessary overhead and complexity.
The library was first written in Python using the Numpy library to fine tune the architecture prior to writing it fully in C++.
Both versions can be found within this repository.

It integrates:
- A dynamic autograd (automatic differentation) engine
- A Tensor API with Eigen-backed numerical kernels
- A small but extensible neural module system (`Linear`, `MLP`)
- Python bindings via `pybind11` for use of library in python code.
- Matched Python and C++ benchmark harnesses for direct comparison

This repo also includes a pure-Python “reference” TinyTorch and benchmark scripts to compare:
- C++ executable
- C++ core called from Python (pybind module)
- pure Python TinyTorch
- PyTorch baseline

## Core Features
1. Dynamic Autograd Engine
   - Each operation creates a dedicated backward node used to calculate gradients during backpropagation.
   - Graph traversal via BLANK similar to PyTorch Library.
   - Supports tensor operations:
        `add`, `mul`, `matmul`, `exp`, `log`, `sigmoid`, `relu`, `pow`, `sum`, `mean`, and more.
2. Tensor API
   - Backed by Eigen, allows high-performance native C++ vectorization.
   - Specialized in small-matrix workloads typical in embedded inference.
3. Modules
   - `Linear(in_features, out_features)`
   - `MLP([layers)` with configurable activation and output function.
   - `.parameters()` returns a structured parameter list compatible with optimizers.
4. Optimizers
   - SGD (stochastic gradient descent) without hidden state.
5. Python Bindings
   - Exposes the full C++ engine to Python with zero copy where possible.
   - Enables training loops that look like PyTorch while executing C++ ops.

## Build prerequisites

- CMake (>= 3.14)
- A C++17 compiler
- Eigen (installed and available on your include path)
- Python 3.12 (if building bindings)
- pybind11 is included as a submodule in `extern/pybind11`

## Build (C++ bench + Python module)

From the repo root:
### 1. Initialize Submodules
```
git submodule update --init --recursive
```

### 2. Build C++ Core + Python Extension
```
cmake -S . -B build
cmake --build build -j
```

This produces:
- `build/tinytorch_cpp_bench.out` – C++ benchmark executable
- `build/tinytorch_cpp.*.so` – Python extension module

## Running the Benchmarks
**C++ Benchmark**
```
./build/tinytorch_cpp_bench.out
```

**Python Calling C++ Core**
```
python3 tinytorch-python/tinytorch_py/tinytorch_cpp_bench.py
```

**Pure Python TinyTorch**
```
python3 tinytorch-python/tinytorch_py/tinytorch_py_bench.py
```

**PyTorch Baseline**
```
python3 tinytorch-python/tinytorch_py/pytorch_bench.py
```

All three follow equivalent workloads:
- Tiny 1-D training demo + linear regression on synthetic 2-D data + MLP regression
- Wall-clock timing for comparison

