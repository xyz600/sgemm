find_package(CUDA REQUIRED)

set(CMAKE_CXX_COMPILER nvcc)
set(CMAKE_CXX_FLAGS "--generate-code arch=compute_61,code=sm_61")

include_directories(${CMAKE_SOURCE_DIR}/matrix)

add_library(cuda_matrix
    matrix_gpu.cpp
    sgemm_kernel.cu
)