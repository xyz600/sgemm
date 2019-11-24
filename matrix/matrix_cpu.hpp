#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgemm_cpu
{
class MatrixCPU
{
public:
    MatrixCPU(std::size_t size);

    void multiply(const MatrixCPU& right, MatrixCPU& out) const noexcept;

    void clear();

    float& value(std::size_t i, std::size_t j) noexcept;
    const float& value(std::size_t i, std::size_t j) const noexcept;

private:
    std::size_t exponential_ceil(const std::size_t) const noexcept;

    float& operator[](const std::size_t index) noexcept;
    const float& operator[](const std::size_t index) const noexcept;

    std::size_t size_;
    std::size_t stride_;
    std::vector<float> data_;
};
} // namespace sgemm_cpu