#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

const float ELU_ALPHA = 1.0f;

__global__ void elu_kernel(const float* input, float* output, int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float x = input[i];
        output[i] = (x > 0.0f) ? x : ELU_ALPHA * (__expf(x) - 1.0f);
    }
}

void cpu_elu(const float* input, float* output, int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        const float x = input[i];
        output[i] = (x > 0.0f) ? x : ELU_ALPHA * (std::exp(x) - 1.0f);
    }
}

bool validate_results(const float* cpu_out, const float* gpu_out, int num_elements, float tolerance = 1e-5) {
    for (int i = 0; i < num_elements; ++i) {
        if (std::fabs(cpu_out[i] - gpu_out[i]) > tolerance) {
            std::cout << "Mismatch at index " << i 
                      << ": CPU=" << cpu_out[i] 
                      << ", GPU=" << gpu_out[i] 
                      << std::endl;
            return false;
        }
    }
    return true;
}
