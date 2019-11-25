include_directories(${CMAKE_SOURCE_DIR}/matrix)

CUDA_ADD_LIBRARY(cuda_matrix STATIC
    matrix_gpu.cu
    sgemm_kernel.cu
)