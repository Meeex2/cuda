#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>  // Include for memcpy

// CUDA Kernel for ReLU²
__global__ void relu2_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = (x > 0) ? x * x : 0.0f;  // ReLU²: x² if x > 0, else 0
    }
}

// CPU Implementation of ReLU² for Comparison
void relu2_cpu(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        float x = data[i];
        data[i] = (x > 0) ? x * x : 0.0f;  // ReLU²: x² if x > 0, else 0
    }
}

// Function to initialize data with random values
void initializeData(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;  // Random values between -0.5 and 0.5
    }
}

// Function to compare two arrays for correctness
bool compareArrays(const float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

