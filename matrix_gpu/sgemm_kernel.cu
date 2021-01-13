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
    const int num_thread = blockDim.x * blockDim.y;

    // thread 単位で small_balock_size^2 だけ要素を持っている時に確保できる block_size_y
    const int block_size_y = num_thread * small_block_size * small_block_size / block_size_x;
    assert(block_size_y <= block_size_x);

    constexpr int block_k_size = 16;
    const int k_lane_idx = thread_idx % block_k_size;
    const int k_warp_idx = thread_idx / block_k_size;
    const int k_warp_per_block = num_thread / block_k_size;

    for (int i = blockIdx.y * block_size_y; i < size; i += gridDim.y * block_size_y)
    {
        for (int j = blockIdx.x * block_size_x; j < size; j += gridDim.x * block_size_x)
        {
            const int base_i = i + threadIdx.y * small_block_size;
            const int base_j = j + threadIdx.x * small_block_size;
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
                {
                    const int l = thread_idx % block_size_x;
                    if (l < width)
                    {
                        for (int kk = thread_idx / block_size_x; kk < block_k_size; kk += num_thread / block_size_x)
                        {
                            temp_b[kk][l] = b[(k + kk) * stride + (j + l)];
                        }
                    }
                }
                {
                    const int kk = k_lane_idx;
                    for (int l = k_warp_idx; l < height; l += k_warp_per_block)
                    {
                        temp_a[kk][l] = a[(i + l) * stride + k + kk];
                    }
                }
                __syncthreads();

                if (has_result)
                {
                    for (int kk = 0; kk < block_k_size; kk++)
                    {
                        for (int ii = 0; ii < small_block_size; ii++)
                        {
                            for (int jj = 0; jj < small_block_size; jj++)
                            {
                                local_result[ii][jj] += temp_a[kk][small_block_size * threadIdx.y + ii] *
                                                        temp_b[kk][small_block_size * threadIdx.x + jj];
                            }
                        }
                    }
                }
                __syncthreads();
            }
            if (has_result)
            {
                for (int ii = 0; ii < small_block_size; ii++)
                {
                    for (int jj = 0; jj < small_block_size; jj++)
                    {
                        result[(base_i + ii) * stride + (base_j + jj)] = local_result[ii][jj];
                    }
                }
            }
            __syncthreads();
        }
    }
}