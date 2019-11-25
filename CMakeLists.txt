cmake_minimum_required(VERSION 3.8)
project(benchmark CXX)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_COMPILER nvcc)
set(CUDA_HOST_COMPILER "clang++")
set(CUDA_NVCC_FLAGS "--generate-code arch=compute_61,code=sm_61 --compiler-options -fPIC")

add_subdirectory(matrix)
add_subdirectory(cuda_matrix)

include_directories(
    ${CMAKE_SOURCE_DIR}/matrix
    ${CMAKE_SOURCE_DIR}/cuda_matrix
)

CUDA_ADD_EXECUTABLE(benchmark
    main.cpp
)

link_directories(
    ${CMAKE_SOURCE_DIR}/matrix
    ${CMAKE_SOURCE_DIR}/cuda_matrix
)

target_link_libraries(benchmark
    matrix
    cuda_matrix
)