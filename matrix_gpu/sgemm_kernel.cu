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
    // blockDim.x == 1024 を前提にしているコード

    constexpr int block_size = 32;

    const int block_per_size = (size / block_size);
    const int num_block = block_per_size * block_per_size;

    const int ii = threadIdx.x / block_size;
    const int jj = threadIdx.x % block_size;
    const int ii_size = blockDim.x / block_size;

    constexpr int block_k_size = 128;
    __shared__ float temp_a[block_size][block_k_size], temp_b_t[block_size][block_k_size];

    for (int block_idx = blockIdx.x; block_idx < num_block; block_idx += gridDim.x)
    {
        const int by = (block_idx / block_per_size) * block_size;
        const int bx = (block_idx % block_per_size) * block_size;

        const int i = by + ii;
        const int j = bx + jj;
        float* temp_result = &result[i * stride + j];

        for (int block_k = 0; block_k < size; block_k += block_k_size)
        {
            // copy a & b from shared memory
            {
                const int k = threadIdx.x % block_k_size;
                for (int ty = threadIdx.x / block_k_size; ty < block_size; ty += blockDim.x / block_k_size)
                {
                    temp_a[ty][k] = a[(by + ty) * stride + block_k + k];
                    temp_b_t[ty][k] = b[(block_k + k) * stride + bx + ty];
                }
            }
            __syncthreads();

            for (int k = 0; k < block_k_size; k++)
            {
                *temp_result += temp_a[ii][k] * temp_b_t[jj][k];
            }
            __syncthreads();
        }
    }
}