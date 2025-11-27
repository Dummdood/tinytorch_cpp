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
  
## Performance Results

As seen in the benchmarks, pure-C++ TinyTorch performed nearly 2× faster than the PyTorch baseline.  
This improvement is likely due to TinyTorch’s lower framework overhead, predictable execution model, and the absence of PyTorch’s substantial dynamic runtime machinery.

When using the library through a Python frontend via pybind11 bindings, the performance gain narrows to roughly 10%.  
This reduction is expected: Python↔C++ crossing introduces measurable overhead, and training loops executed on the Python side require frequent cross-language calls, partially offsetting TinyTorch’s lightweight C++ execution.

In addition, pure-C++ TinyTorch and pure-Python TinyTorch (NumPy-based) perform similarly.  
Reasons likely include:

- The Python implementation uses NumPy, which is backed by highly optimized binaries.  
- The C++ version makes heavy use of `shared_ptr` allocations to construct the autograd graph, which becomes a dominant cost for small tensors.  

Overall, the results validate the design goal:  
TinyTorch achieves lower per-operation overhead, predictable execution, and strong C++ performance, while remaining interoperable with Python when needed.

  
## Lessons Learned

Building TinyTorch-CPP provided experience with:
- Designing a computation graph and writing backward passes manually.
- Debugging gradient flows and validating autodiff correctness.
- Efficient C++ memory ownership (shared_ptr, weak_ptr).
- Isolating graph nodes and avoiding memory leaks.
- Using Eigen effectively for linear algebra.
- Exposing a full C++ API to Python using pybind11.
- Writing fair benchmark harnesses comparing Python, C++, and PyTorch.
- Understanding why real ML frameworks use heavy vectorization and fused kernels.

## Future Improvements
- Broadcasting support
- Batch training utilities
- More activation and loss functions (tanh, cross-entropy)
- LayerNorm, Softmax, Dropout
- Parameter serialization (state_dict-like)
- Memory pool for graph allocations
- Fused kernels and more aggressive Eigen optimization

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

