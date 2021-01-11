#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

#include "matrix_cpu.hpp"
#include "matrix_gpu.cuh"

template <typename T>
void prepare_input(T& matrix1, T& matrix2)
{
    assert(false);
}

template <>
void prepare_input(MatrixCPU& matrix1, MatrixCPU& matrix2)
{
    assert(matrix1.size() == matrix2.size());

    std::mt19937_64 mt(0);
    std::uniform_real_distribution<float> rand;

    for (int i = 0; i < matrix1.size(); i++)
    {
        for (int j = 0; j < matrix1.size(); j++)
        {
            matrix1.value(i, j) = rand(mt);
            matrix2.value(i, j) = rand(mt);
        }
    }
}

template <>
void prepare_input(MatrixGPU& matrix1, MatrixGPU& matrix2)
{
    assert(matrix1.size() == matrix2.size());

    MatrixCPU mat1(matrix1.size()), mat2(matrix1.size());
    prepare_input(mat1, mat2);

    matrix1.copy_from(mat1);
    matrix2.copy_from(mat2);
}

void check_cpu()
{
    constexpr std::size_t small_size = 128;
    // TEST
    MatrixCPU m1(small_size);
    MatrixCPU m2(small_size);
    prepare_input(m1, m2);

    MatrixCPU result1(small_size);
    MatrixCPU result2(small_size);

    m1.multiply(m2, result1);
    m1.multiply_base(m2, result2);

    for (int i = 0; i < small_size; i++)
    {
        for (int j = 0; j < small_size; j++)
        {
            const auto v1 = result1.value(i, j);
            const auto v2 = result2.value(i, j);
            if (abs(v1 - v2) / v1 > 1e-4)
            {
                std::cerr << v1 << ", " << v2 << std::endl;
                std::cerr << "test failed..." << std::endl;
                exit(1);
            }
        }
    }
    std::cerr << "test passed." << std::endl;
}

void check_gpu()
{
    constexpr std::size_t small_size = 128;
    // TEST
    MatrixCPU m1(small_size);
    MatrixCPU m2(small_size);
    prepare_input(m1, m2);

    MatrixCPU result1(small_size);
    m1.multiply_base(m2, result1);

    MatrixGPU m1_dev(small_size);
    MatrixGPU m2_dev(small_size);
    m1_dev.copy_from(m1);
    m2_dev.copy_from(m2);

    MatrixGPU result_dev(small_size);
    m1_dev.multiply(m2_dev, result_dev);

    MatrixCPU result2(small_size);
    result_dev.copy_to(result2);

    for (int i = 0; i < small_size; i++)
    {
        for (int j = 0; j < small_size; j++)
        {
            const auto v1 = result1.value(i, j);
            const auto v2 = result2.value(i, j);
            if (abs(v1 - v2) / v1 > 1e-4)
            {
                std::cerr << v1 << ", " << v2 << std::endl;
                std::cerr << "test failed..." << std::endl;
                exit(1);
            }
        }
    }
    std::cerr << "test passed." << std::endl;
}

template <typename T>
std::size_t experiment(const std::size_t size, const std::size_t iteration)
{
    T matrix1(size);
    T matrix2(size);

    prepare_input(matrix1, matrix2);

    T result(size);
    std::size_t elapsed = 0;

    for (std::size_t i = 0; i < iteration; i++)
    {
        result.clear();
        auto start = std::chrono::system_clock::now();
        matrix1.multiply(matrix2, result);
        auto end = std::chrono::system_clock::now();
        elapsed += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return elapsed / iteration;
}

int main(int argc, char* argv[])
{
    constexpr std::size_t size = 2048;
    constexpr std::size_t iteration = 10;

    const auto to_gflops = [&](std::size_t elapsed) {
        return static_cast<double>(2 * size * size * size / 1.0e9) / (elapsed * 1e-6);
    };

    {
        std::cout << "CPU experiment" << std::endl;
        check_cpu();
        const auto average_elapsed_cpu = experiment<MatrixCPU>(size, iteration);
        std::cout << "average cpu time: " << average_elapsed_cpu << "[us]" << std::endl;
        const auto cpu_ideal_gflops = (4.3 * 8 * 12 * 2);
        const auto gflops = to_gflops(average_elapsed_cpu);
        std::cout << "Gflops: " << gflops << std::endl;
        std::cout << "ideal Gflops: " << cpu_ideal_gflops << std::endl;
        std::cout << "rate: " << (gflops / cpu_ideal_gflops) << std::endl;
        std::cout << "=========" << std::endl;
    }

    {
        std::cout << "GPU experiment" << std::endl;
        check_gpu();
        const auto average_elapsed_gpu = experiment<MatrixGPU>(size, iteration);
        std::cout << "average gpu time: " << average_elapsed_gpu << "[us]" << std::endl;
        const auto gflops = to_gflops(average_elapsed_gpu);
        std::cout << "Gflops: " << gflops << std::endl;
        std::cout << "=========" << std::endl;
    }

    return 0;
}