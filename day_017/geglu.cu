#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

__global__ void geglu_kernel(const float* input, float* output, int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements / 2) return;
    const int half_dim = num_elements / 2;
    const float x = input[idx];
    const float y = input[idx + half_dim];
    
    
    const float gelu = x * 0.5f * (1.0f + tanhf(1.702f * x));
    
    output[idx] = gelu * y;
}

void geglu_cpu(const float* input, float* output, int num_elements) {
    const int half_dim = num_elements / 2;
    for (int i = 0; i < half_dim; ++i) {
        const float x = input[i];
        const float y = input[i + half_dim];
        
        
        const float gelu = x * 0.5f * (1.0f + std::tanh(1.702f * x));
        
        output[i] = gelu * y;
    }
}

