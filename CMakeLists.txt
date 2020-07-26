cmake_minimum_required(VERSION 3.8)
project(benchmark CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER clang++)
# set(CUDA_HOST_COMPILER "clang++")
# set(CUDA_NVCC_FLAGS "--generate-code arch=compute_61,code=sm_61 --compiler-options -fPIC")

set(CMAKE_CXX_COMPILE_FEATURES "-O2 -Werror")

add_subdirectory(matrix)
# add_subdirectory(cuda_matrix)

include_directories(
    ${CMAKE_SOURCE_DIR}/matrix
    # ${CMAKE_SOURCE_DIR}/cuda_matrix
)

CUDA_ADD_EXECUTABLE(benchmark
    main.cpp
)

link_directories(
    ${CMAKE_SOURCE_DIR}/matrix
    # ${CMAKE_SOURCE_DIR}/cuda_matrix
)

target_link_libraries(benchmark
    matrix
    # cuda_matrix
)