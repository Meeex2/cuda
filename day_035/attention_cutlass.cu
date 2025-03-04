#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

__global__ void softmax_kernel(const float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float max_val = input[row * num_cols];
        for (int col = 1; col < num_cols; col++) {
            max_val = fmaxf(max_val, input[row * num_cols + col]);
        }
        float sum = 0.0f;
        for (int col = 0; col < num_cols; col++) {
            output[row * num_cols + col] = expf(input[row * num_cols + col] - max_val);
            sum += output[row * num_cols + col];
        }
        for (int col = 0; col < num_cols; col++) {
            output[row * num_cols + col] /= sum;
        }
    }
}

__global__ void weighted_aggregation_kernel(const float* attention_weights, const float* values, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (int col = 0; col < num_cols; col++) {
            float sum = 0.0f;
            for (int k = 0; k < num_cols; k++) {
                sum += attention_weights[row * num_cols + k] * values[k * num_cols + col];
            }
            output[row * num_cols + col] = sum;
        }
    }
}


