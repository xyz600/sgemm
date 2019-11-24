#pragma once

#include <cstddef>

class MatrixCPU;

class MatrixGPU
{
public:
    __host__ MatrixGPU(std::size_t size);

    __host__ ~MatrixGPU();

    __host__ void multiply(const MatrixGPU& right, MatrixGPU& out) const noexcept;

    __host__ void copy_from(const MatrixCPU& src) noexcept;

    __host__ void copy_to(MatrixCPU& dst) const noexcept;

    __host__ void clear() noexcept;

    __host__ std::size_t size() const noexcept;

private:
    std::size_t exponential_ceil(const std::size_t size) const noexcept;

    float* data_;

    std::size_t size_;

    std::size_t stride_;
};