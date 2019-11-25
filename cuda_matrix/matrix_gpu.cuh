#pragma once

#include <cstddef>

class MatrixCPU;

class MatrixGPU
{
public:
    MatrixGPU(std::size_t size);

    ~MatrixGPU();

    void multiply(const MatrixGPU& right, MatrixGPU& out) const noexcept;

    void copy_from(const MatrixCPU& src) noexcept;

    void copy_to(MatrixCPU& dst) const noexcept;

    void clear() noexcept;

    std::size_t size() const noexcept;

private:
    std::size_t exponential_ceil(const std::size_t size) const noexcept;

    float* data_;

    std::size_t size_;

    std::size_t stride_;
};