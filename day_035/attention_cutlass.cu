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

void self_attention_cpu(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& output, int num_rows, int num_cols) {
    std::vector<float> scores(num_rows * num_cols, 0.0f);
    std::vector<float> attention_weights(num_rows * num_cols, 0.0f);
    
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_cols; k++) {
                sum += Q[i * num_cols + k] * K[j * num_cols + k];
            }
            scores[i * num_cols + j] = sum / sqrtf(num_cols);
        }
    }
    
    for (int i = 0; i < num_rows; i++) {
        float max_val = scores[i * num_cols];
        for (int j = 1; j < num_cols; j++) {
            max_val = fmaxf(max_val, scores[i * num_cols + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < num_cols; j++) {
            attention_weights[i * num_cols + j] = expf(scores[i * num_cols + j] - max_val);
            sum += attention_weights[i * num_cols + j];
        }
        for (int j = 0; j < num_cols; j++) {
            attention_weights[i * num_cols + j] /= sum;
        }
    }
    
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_cols; k++) {
                sum += attention_weights[i * num_cols + k] * V[k * num_cols + j];
            }
            output[i * num_cols + j] = sum;
        }
    }
}

