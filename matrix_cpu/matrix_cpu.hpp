#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

class MatrixCPU
{
public:
    MatrixCPU(std::size_t size);

    void multiply(const MatrixCPU& right, MatrixCPU& out) const noexcept;

    void multiply_base(const MatrixCPU& right, MatrixCPU& out) const noexcept;

    void clear();

    float& value(std::size_t i, std::size_t j) noexcept;
    const float& value(std::size_t i, std::size_t j) const noexcept;

    std::size_t size() const noexcept;

    const float* data() const noexcept;

private:
    std::size_t exponential_ceil(const std::size_t) const noexcept;

    float& operator[](const std::size_t index) noexcept;
    const float& operator[](const std::size_t index) const noexcept;

    void kernel_4x24(const MatrixCPU& right, MatrixCPU& out, std::size_t ii, std::size_t jj,
                     std::size_t kk) const noexcept;

    void kernel_4x16(const MatrixCPU& right, MatrixCPU& out, std::size_t ii, std::size_t jj,
                     std::size_t kk) const noexcept;

    static constexpr std::size_t block_size = 64;

    std::size_t size_;
    std::size_t stride_;
    std::vector<float> data_;
};