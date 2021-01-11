#pragma once

#include <cstddef>

class MatrixCPU;

class MatrixGPU
{
public:
    MatrixGPU(int size);

    ~MatrixGPU();

    void multiply(const MatrixGPU& right, MatrixGPU& out) const noexcept;

    void copy_from(const MatrixCPU& src) noexcept;

    void copy_to(MatrixCPU& dst) const noexcept;

    void clear() noexcept;

    std::size_t size() const noexcept;

private:
    std::size_t exponential_ceil(const int size) const noexcept;

    float* data_;

    int size_;

    int stride_;

    int grid_size_;

    int block_size_;
};