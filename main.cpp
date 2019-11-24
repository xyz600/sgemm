#include <chrono>
#include <iostream>
#include <random>

#include "matrix_cpu.hpp"

std::size_t experiment_cpu(const std::size_t size, const std::size_t iteration)
{
    sgemm_cpu::MatrixCPU matrix1(size);
    sgemm_cpu::MatrixCPU matrix2(size);
    sgemm_cpu::MatrixCPU result(size);

    std::mt19937_64 mt(0);
    std::uniform_real_distribution<float> rand;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix1.value(i, j) = rand(mt);
            matrix2.value(i, j) = rand(mt);
        }
    }

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
    constexpr std::size_t size = 512;
    constexpr std::size_t iteration = 1;

    const auto average_elapsed = experiment_cpu(size, iteration);

    std::cout << "average time: " << average_elapsed << "[us]" << std::endl;

    return 0;
}