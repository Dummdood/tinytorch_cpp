cmake_minimum_required(VERSION 3.14)
project(TinyTorchCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Your headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Eigen – you were using -I /opt/homebrew/... manually before
include_directories(/opt/homebrew/opt/eigen/include/eigen3)

# -------------------------------------------------------------------
# 1) Add pybind11 from the submodule
# -------------------------------------------------------------------
add_subdirectory(extern/pybind11)

# -------------------------------------------------------------------
# 2) Core library with all TinyTorch sources
# -------------------------------------------------------------------
add_library(tinytorch_core
    src/tensor.cpp
    src/backward_nodes.cpp
    src/nn.cpp
    src/loss.cpp
)

target_include_directories(tinytorch_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Make sure the core library is position-independent so it can be linked into the Python module
set_property(TARGET tinytorch_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------------------
# 3) C++ demo executable
# -------------------------------------------------------------------
add_executable(tinytorch_cpp_bench.out
    src/tinytorch_cpp_bench.cpp
)

target_link_libraries(tinytorch_cpp_bench.out PRIVATE tinytorch_core)

# -------------------------------------------------------------------
# 4) Python extension module using pybind11
# -------------------------------------------------------------------
pybind11_add_module(tinytorch_cpp
    src/bindings.cpp
)

target_link_libraries(tinytorch_cpp PRIVATE tinytorch_core)

target_include_directories(tinytorch_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)