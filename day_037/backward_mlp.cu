#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// CUDA kernel for ReLU derivative
__global__ void relu_derivative_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0) ? 1.0f : 0.0f;
    }
}

// CUDA kernel for computing gradients of the output layer
__global__ void output_layer_gradients_kernel(const float* output, const int* labels, float* grad_output, int size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int sample_idx = idx / num_classes;
        int label = labels[sample_idx];
        grad_output[idx] = output[idx] - (idx % num_classes == label ? 1.0f : 0.0f);
    }
}


