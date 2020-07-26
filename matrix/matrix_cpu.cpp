#include "matrix_cpu.hpp"

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cassert>

std::size_t MatrixCPU::exponential_ceil(const std::size_t size) const noexcept { return (size + size - 1) / 2 * 2; }

MatrixCPU::MatrixCPU(const std::size_t size) : size_(size), stride_(exponential_ceil(size))
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

void MatrixCPU::multiply_fast(const MatrixCPU& right, MatrixCPU& out) const noexcept
{
    assert(right.size_ == size_);
    assert(size_ == out.size_);

    constexpr std::size_t block_size = 64;

#pragma omp parallel for
    for (std::size_t i = 0; i < size_; i += block_size)
    {
        for (std::size_t k = 0; k < size_; k += block_size)
        {
            for (std::size_t j = 0; j < size_; j += block_size)
            {
                for (std::size_t ii = i; ii < i + block_size; ii++)
                {
                    for (std::size_t kk = k; kk < k + block_size; kk++)
                    {
                        for (std::size_t jj = j; jj < j + block_size; jj++)
                        {
                            out[ii * stride_ + jj] += data_[ii * stride_ + kk] * right.data_[kk * stride_ + jj];
                        }
                    }
                }
            }
        }
    }
}

void MatrixCPU::clear() { std::fill(data_.begin(), data_.end(), 0.0f); }

float& MatrixCPU::value(std::size_t i, std::size_t j) noexcept { return data_[i * stride_ + j]; }

const float& MatrixCPU::value(std::size_t i, std::size_t j) const noexcept { return data_[i * stride_ + j]; }

std::size_t MatrixCPU::size() const noexcept { return size_; }

const float* MatrixCPU::data() const noexcept { return data_.data(); }