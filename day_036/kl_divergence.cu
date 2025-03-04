#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// CUDA kernel for KL divergence
__global__ void kl_divergence_kernel(const float* P, const float* Q, float* divergence, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = P[idx];
        float q = Q[idx];
        divergence[idx] = p * logf(p / q);
    }
}

// CPU reference implementation of KL divergence
float kl_divergence_cpu(const std::vector<float>& P, const std::vector<float>& Q, int size) {
    float divergence = 0.0f;
    for (int i = 0; i < size; i++) {
        divergence += P[i] * logf(P[i] / Q[i]);
    }
    return divergence;
}

