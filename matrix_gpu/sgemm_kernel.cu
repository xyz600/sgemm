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

    constexpr int block_k_size = 128;
    __shared__ float temp_a[block_k_size][block_size], temp_b_t[block_k_size][block_size];

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
                const auto* global_temp_a = a + block_k + k;
                for (int ty = threadIdx.x / block_k_size; ty < block_size; ty += blockDim.x / block_k_size)
                {
                    temp_a[k][ty] = global_temp_a[(by + ty) * stride];
                }
            }
            {
                const int ty = threadIdx.x % block_size;
                const auto* global_temp_b = b + bx + ty;
                for (int k = threadIdx.x / block_size; k < block_k_size; k += blockDim.x / block_size)
                {
                    temp_b_t[k][ty] = global_temp_b[(block_k + k) * stride];
                }
            }
            __syncthreads();

            for (int k = 0; k < block_k_size; k++)
            {
                *temp_result += temp_a[k][ii] * temp_b_t[k][jj];
            }
            __syncthreads();
        }
    }
}