cmake_minimum_required(VERSION 3.14)
project(TinyTorchCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core include directory for your headers (tensor.hpp, linear.hpp, etc.)
include_directories(${PROJECT_SOURCE_DIR}/include)

# -------------------------------------------------------------------
# 1) Add pybind11 from the submodule
# -------------------------------------------------------------------
add_subdirectory(extern/pybind11)

# -------------------------------------------------------------------
# 2) C++ demo executable (your existing main.cpp)
# -------------------------------------------------------------------
add_executable(tinytorch_demo
    src/main.cpp
)

# If main.cpp uses your headers only (header-only framework),
# you don't need to link anything extra here.

# -------------------------------------------------------------------
# 3) Python extension module using pybind11
#    This will build a module named `tinytorch_cpp` importable from Python.
# -------------------------------------------------------------------
pybind11_add_module(tinytorch_cpp
    src/bindings.cpp
)

target_include_directories(tinytorch_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

