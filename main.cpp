#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

#include "matrix_cpu.hpp"

template <typename T> void prepare_input(T& matrix1, T& matrix2) { assert(false); }

template <> void prepare_input(MatrixCPU& matrix1, MatrixCPU& matrix2)
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

template <typename T> std::size_t experiment(const std::size_t size, const std::size_t iteration)
{
    T matrix1(size);
    T matrix2(size);
    T result(size);

    prepare_input(matrix1, matrix2);

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

    const auto average_elapsed_cpu = experiment<MatrixCPU>(size, iteration);
    std::cout << "average cpu time: " << average_elapsed_cpu << "[us]" << std::endl;

    // const auto average_elapsed_gpu = experiment<MatrixGPU>(size, iteration);
    // std::cout << "average gpu time: " << average_elapsed_gpu << "[us]" << std::endl;

    return 0;
}