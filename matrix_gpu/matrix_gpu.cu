#include <cuda_runtime.h>
#include <cassert>

#include "matrix_cpu.hpp"
#include "matrix_gpu.cuh"
#include "sgemm_kernel.cuh"

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d,  ", __FILE__, __LINE__);                     \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

MatrixGPU::MatrixGPU(int size) : size_(size), stride_(exponential_ceil(size_))
{
    CHECK(cudaMalloc((void**)&data_, sizeof(float) * size_ * stride_));
    CHECK(cudaOccupancyMaxPotentialBlockSize(&grid_size_, &block_size_, sgemm));
    block_size_ = block_size_ / 32 * 32;
}

MatrixGPU::~MatrixGPU() { cudaFree(data_); }

std::size_t MatrixGPU::exponential_ceil(const int size) const noexcept { return (size + size - 1) / 2 * 2; }

void MatrixGPU::multiply(const MatrixGPU& right, MatrixGPU& out) const noexcept
{
    assert(size_ == right.size_);
    assert(size_ == out.size_);
    sgemm<<<grid_size_, block_size_>>>(data_, right.data_, out.data_, size_, stride_);
}

void MatrixGPU::copy_from(const MatrixCPU& src) noexcept
{
    CHECK(cudaMemcpy((void*)data_, (void*)src.data(), sizeof(float) * stride_ * size_, cudaMemcpyHostToDevice));
}

void MatrixGPU::copy_to(MatrixCPU& dst) const noexcept
{
    CHECK(cudaMemcpy((void*)dst.data(), (void*)data_, sizeof(float) * stride_ * size_, cudaMemcpyDeviceToHost));
}

void MatrixGPU::clear() noexcept { fill<<<grid_size_, block_size_>>>(data_, size_, 0); }

std::size_t MatrixGPU::size() const noexcept { return size_; }