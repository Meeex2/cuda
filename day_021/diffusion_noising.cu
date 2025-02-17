#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

__global__ void diffusion_noise_kernel(float* x, const float* noise, const float* alpha_bar, int timestep, int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float sqrt_alpha = sqrtf(alpha_bar[timestep]);
        const float sqrt_one_minus_alpha = sqrtf(1.0f - alpha_bar[timestep]);
        x[i] = sqrt_alpha * x[i] + sqrt_one_minus_alpha * noise[i];
    }
}

void cpu_diffusion_noise(float* x, const float* noise, const float* alpha_bar, int timestep, int num_elements) {
    const float sqrt_alpha = sqrt(alpha_bar[timestep]);
    const float sqrt_one_minus_alpha = sqrt(1.0f - alpha_bar[timestep]);
    
    for (int i = 0; i < num_elements; ++i) {
        x[i] = sqrt_alpha * x[i] + sqrt_one_minus_alpha * noise[i];
    }
}


