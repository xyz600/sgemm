#pragma once

constexpr int small_block_size = 4;

constexpr int block_size_x = 128;

__global__ void fill(float* data, int size, float value);

__global__ void sgemm(const float* a, const float* b, float* result, int size, int stride);