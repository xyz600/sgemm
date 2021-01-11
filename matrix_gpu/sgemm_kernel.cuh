#pragma once

__global__ void fill(float* data, int size, float value);

__global__ void sgemm(const float* a, const float* b, float* result, int size, int stride);