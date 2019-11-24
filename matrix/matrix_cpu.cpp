#include "matrix_cpu.hpp"

#include <algorithm>
#include <cassert>

std::size_t MatrixCPU::exponential_ceil(const std::size_t size) const noexcept { return (size + size - 1) / 2 * 2; }

MatrixCPU::MatrixCPU(const std::size_t size)
    : size_(size)
    , stride_(exponential_ceil(size))
{
    data_.resize(size_ * stride_);
}

float& MatrixCPU::operator[](const std::size_t index) noexcept { return data_[index]; }
const float& MatrixCPU::operator[](const std::size_t index) const noexcept { return data_[index]; }

void MatrixCPU::multiply(const MatrixCPU& right, MatrixCPU& out) const noexcept
{
    assert(right.size_ == size_);
    assert(size_ == out.size_);

    for (std::size_t i = 0; i < size_; i++)
    {
        for (std::size_t j = 0; j < size_; j++)
        {
            for (std::size_t k = 0; k < size_; k++)
            {
                out[i * stride_ + j] += data_[i * stride_ + k] * data_[k * stride_ + j];
            }
        }
    }
}

void MatrixCPU::clear() { std::fill(data_.begin(), data_.end(), 0.0f); }

float& MatrixCPU::value(std::size_t i, std::size_t j) noexcept { return data_[i * stride_ + j]; }

const float& MatrixCPU::value(std::size_t i, std::size_t j) const noexcept { return data_[i * stride_ + j]; }

std::size_t MatrixCPU::size() const noexcept { return size_; }

const float* MatrixCPU::data() const noexcept { return data_.data(); }