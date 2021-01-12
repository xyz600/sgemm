#include <cassert>
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
    const int thread_idx = threadIdx.y * warpSize + threadIdx.x;
    const int lane_idx = thread_idx % warpSize;
    const int warp_idx = thread_idx / warpSize;
    const int warp_per_block = blockDim.x / warpSize;
    const int num_thread = blockDim.x * blockDim.y;

    // thread 単位で small_balock_size^2 だけ要素を持っている時に確保できる block_size_y
    const int block_size_y = num_thread * small_block_size * small_block_size / block_size_x;
    assert(block_size_y <= block_size_x);

    constexpr int block_k_size = 32;

    for (int i = blockIdx.y * block_size_y; i < size; i += gridDim.y * block_size_y)
    {
        for (int j = blockIdx.x * block_size_x; j < size; j += gridDim.x * block_size_x)
        {
            const int base_i = i + threadIdx.y * 2;
            const int base_j = j + threadIdx.x * 2;
            const bool has_result = (base_i < size && base_j < size);

            // 単一スレッドの結果保存用
            float local_result[small_block_size][small_block_size];
            for (int ii = 0; ii < small_block_size; ii++)
            {
                for (int jj = 0; jj < small_block_size; jj++)
                {
                    local_result[ii][jj] = 0;
                }
            }

            const int height = min(block_size_y, size - i);
            const int width = min(block_size_x, size - j);

            for (int k = 0; k < size; k += block_k_size)
            {
                __shared__ float temp_a[block_k_size][block_size_x + 1], temp_b[block_k_size][block_size_x + 1];
                for (int kk = 0; kk < block_k_size; kk++)
                {
                    for (int l = thread_idx; l < width; l += num_thread)
                    {
                        temp_b[kk][l] = b[(k + kk) * stride + (j + l)];
                    }
                }
                {
                    const int kk = lane_idx;
                    for (int l = warp_idx; l < height; l += warp_per_block)
                    {
                        temp_a[kk][l] = a[(i + l) * stride + k + kk];
                    }
                }
                __syncthreads();

                if (has_result)
                {
                    for (int kk = 0; kk < block_k_size; kk++)
                    {
                        local_result[0][0] += temp_a[kk][small_block_size * threadIdx.y + 0] *
                                              temp_b[kk][small_block_size * threadIdx.x + 0];
                        local_result[0][1] += temp_a[kk][small_block_size * threadIdx.y + 0] *
                                              temp_b[kk][small_block_size * threadIdx.x + 1];
                        local_result[1][0] += temp_a[kk][small_block_size * threadIdx.y + 1] *
                                              temp_b[kk][small_block_size * threadIdx.x + 0];
                        local_result[1][1] += temp_a[kk][small_block_size * threadIdx.y + 1] *
                                              temp_b[kk][small_block_size * threadIdx.x + 1];
                    }
                }
                __syncthreads();
            }
            if (has_result)
            {
                result[base_i * stride + base_j] = local_result[0][0];
                result[base_i * stride + (base_j + 1)] = local_result[0][1];
                result[(base_i + 1) * stride + base_j] = local_result[1][0];
                result[(base_i + 1) * stride + (base_j + 1)] = local_result[1][1];
            }
            __syncthreads();
        }
    }
}