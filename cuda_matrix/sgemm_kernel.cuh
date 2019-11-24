#pragma once

__global__ void fill(float* data, const std::size_t size, const float value);

__global__ void sgemm(const float* a, const float* b, float* result, const std::size_t size, const std::size_t stride);