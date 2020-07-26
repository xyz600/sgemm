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
                for (std::size_t ii = i; ii < i + block_size; ii += 8)
                {
                    for (std::size_t jj = j; jj < j + block_size; jj += 16)
                    {
                        auto op = &out[ii * stride_ + jj];
                        auto o1 = _mm256_loadu_ps(op);
                        auto o2 = _mm256_loadu_ps(op + 8);

                        auto opn = &out[(ii + 1) * stride_ + jj];
                        auto o1n = _mm256_loadu_ps(opn);
                        auto o2n = _mm256_loadu_ps(opn + 8);

                        auto opn2 = &out[(ii + 2) * stride_ + jj];
                        auto o1n2 = _mm256_loadu_ps(opn2);
                        auto o2n2 = _mm256_loadu_ps(opn2 + 8);

                        auto opn3 = &out[(ii + 3) * stride_ + jj];
                        auto o1n3 = _mm256_loadu_ps(opn3);
                        auto o2n3 = _mm256_loadu_ps(opn3 + 8);

                        auto opn4 = &out[(ii + 4) * stride_ + jj];
                        auto o1n4 = _mm256_loadu_ps(opn4);
                        auto o2n4 = _mm256_loadu_ps(opn4 + 8);

                        auto opn5 = &out[(ii + 5) * stride_ + jj];
                        auto o1n5 = _mm256_loadu_ps(opn5);
                        auto o2n5 = _mm256_loadu_ps(opn5 + 8);

                        auto opn6 = &out[(ii + 6) * stride_ + jj];
                        auto o1n6 = _mm256_loadu_ps(opn6);
                        auto o2n6 = _mm256_loadu_ps(opn6 + 8);

                        auto opn7 = &out[(ii + 7) * stride_ + jj];
                        auto o1n7 = _mm256_loadu_ps(opn7);
                        auto o2n7 = _mm256_loadu_ps(opn7 + 8);

                        for (std::size_t kk = k; kk < k + block_size; kk++)
                        {
                            auto i1 = _mm256_set1_ps(data_[ii * stride_ + kk]);
                            auto i1n = _mm256_set1_ps(data_[(ii + 1) * stride_ + kk]);
                            auto i1n2 = _mm256_set1_ps(data_[(ii + 2) * stride_ + kk]);
                            auto i1n3 = _mm256_set1_ps(data_[(ii + 3) * stride_ + kk]);
                            auto i1n4 = _mm256_set1_ps(data_[(ii + 4) * stride_ + kk]);
                            auto i1n5 = _mm256_set1_ps(data_[(ii + 5) * stride_ + kk]);
                            auto i1n6 = _mm256_set1_ps(data_[(ii + 6) * stride_ + kk]);
                            auto i1n7 = _mm256_set1_ps(data_[(ii + 7) * stride_ + kk]);

                            auto ip = &right.data_[kk * stride_ + jj];

                            auto i21 = _mm256_loadu_ps(ip);
                            auto i22 = _mm256_loadu_ps(ip + 8);

                            o1 = _mm256_fmadd_ps(i1, i21, o1);
                            o2 = _mm256_fmadd_ps(i1, i22, o2);

                            o1n = _mm256_fmadd_ps(i1n, i21, o1n);
                            o2n = _mm256_fmadd_ps(i1n, i22, o2n);

                            o1n2 = _mm256_fmadd_ps(i1n2, i21, o1n2);
                            o2n2 = _mm256_fmadd_ps(i1n2, i22, o2n2);

                            o1n3 = _mm256_fmadd_ps(i1n3, i21, o1n3);
                            o2n3 = _mm256_fmadd_ps(i1n3, i22, o2n3);

                            o1n4 = _mm256_fmadd_ps(i1n4, i21, o1n4);
                            o2n4 = _mm256_fmadd_ps(i1n4, i22, o2n4);

                            o1n5 = _mm256_fmadd_ps(i1n5, i21, o1n5);
                            o2n5 = _mm256_fmadd_ps(i1n5, i22, o2n5);

                            o1n6 = _mm256_fmadd_ps(i1n6, i21, o1n6);
                            o2n6 = _mm256_fmadd_ps(i1n6, i22, o2n6);

                            o1n7 = _mm256_fmadd_ps(i1n7, i21, o1n7);
                            o2n7 = _mm256_fmadd_ps(i1n7, i22, o2n7);
                        }

                        _mm256_storeu_ps(op, o1);
                        _mm256_storeu_ps(op + 8, o2);

                        _mm256_storeu_ps(opn, o1n);
                        _mm256_storeu_ps(opn + 8, o2n);

                        _mm256_storeu_ps(opn2, o1n2);
                        _mm256_storeu_ps(opn2 + 8, o2n2);

                        _mm256_storeu_ps(opn3, o1n3);
                        _mm256_storeu_ps(opn3 + 8, o2n3);

                        _mm256_storeu_ps(opn4, o1n4);
                        _mm256_storeu_ps(opn4 + 8, o2n4);

                        _mm256_storeu_ps(opn5, o1n5);
                        _mm256_storeu_ps(opn5 + 8, o2n5);

                        _mm256_storeu_ps(opn6, o1n6);
                        _mm256_storeu_ps(opn6 + 8, o2n6);

                        _mm256_storeu_ps(opn7, o1n7);
                        _mm256_storeu_ps(opn7 + 8, o2n7);
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