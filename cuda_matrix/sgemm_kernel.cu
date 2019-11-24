#include "sgemm_kernel.cuh"

__global__ void fill(float* data, const std::size_t size, const float value)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < size)
    {
        data[idx] = value;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void sgemm(const float* a, const float* b, float* result, const std::size_t size, const std::size_t stride)
{
}