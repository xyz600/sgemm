#include <cstdio>
#include "sgemm_kernel.cuh"

__global__ void fill(float* data, const int size, const float value)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < size)
    {
        data[idx] = value;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void sgemm(const float* a, const float* b, float* result, const int size, const int stride)
{
    for (int i = blockIdx.x; i < size; i += gridDim.x)
    {
        for (int j = threadIdx.x; j < size; j += blockDim.x)
        {
            for (std::size_t k = 0; k < size; k++)
            {
                result[i * stride + j] += a[i * stride + k] * b[k * stride + j];
            }
        }
    }
}