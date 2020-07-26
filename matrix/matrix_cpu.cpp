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
                out[i * stride_ + j] += data_[i * stride_ + k] * right.data_[k * stride_ + j];
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
                    auto op = &out[ii * stride_ + j];
                    auto o1 = _mm256_loadu_ps(op);
                    auto o2 = _mm256_loadu_ps(op + 8);
                    auto o3 = _mm256_loadu_ps(op + 16);
                    auto o4 = _mm256_loadu_ps(op + 24);
                    auto o5 = _mm256_loadu_ps(op + 32);
                    auto o6 = _mm256_loadu_ps(op + 40);
                    auto o7 = _mm256_loadu_ps(op + 48);
                    auto o8 = _mm256_loadu_ps(op + 56);

                    for (std::size_t kk = k; kk < k + block_size; kk++)
                    {
                        auto i1 = _mm256_set1_ps(data_[ii * stride_ + kk]);

                        auto ip = &right.data_[kk * stride_ + j];

                        auto i21 = _mm256_loadu_ps(ip);
                        auto i22 = _mm256_loadu_ps(ip + 8);
                        auto i23 = _mm256_loadu_ps(ip + 16);
                        auto i24 = _mm256_loadu_ps(ip + 24);
                        auto i25 = _mm256_loadu_ps(ip + 32);
                        auto i26 = _mm256_loadu_ps(ip + 40);
                        auto i27 = _mm256_loadu_ps(ip + 48);
                        auto i28 = _mm256_loadu_ps(ip + 56);

                        o1 = _mm256_fmadd_ps(i1, i21, o1);
                        o2 = _mm256_fmadd_ps(i1, i22, o2);
                        o3 = _mm256_fmadd_ps(i1, i23, o3);
                        o4 = _mm256_fmadd_ps(i1, i24, o4);
                        o5 = _mm256_fmadd_ps(i1, i25, o5);
                        o6 = _mm256_fmadd_ps(i1, i26, o6);
                        o7 = _mm256_fmadd_ps(i1, i27, o7);
                        o8 = _mm256_fmadd_ps(i1, i28, o8);
                    }

                    op = &out[ii * stride_ + j];

                    _mm256_storeu_ps(op, o1);
                    _mm256_storeu_ps(op + 8, o2);
                    _mm256_storeu_ps(op + 16, o3);
                    _mm256_storeu_ps(op + 24, o4);
                    _mm256_storeu_ps(op + 32, o5);
                    _mm256_storeu_ps(op + 40, o6);
                    _mm256_storeu_ps(op + 48, o7);
                    _mm256_storeu_ps(op + 56, o8);
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