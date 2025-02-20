#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

__global__ void adam_kernel(
    float* params, 
    const float* grads,
    float* m, 
    float* v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    int num_elements
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];
        
        
        float m_hat = m[i] / (1 - powf(beta1, timestep));
        float v_hat = v[i] / (1 - powf(beta2, timestep));
        
        
        params[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

void cpu_adam(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    int num_elements
) {
    for (int i = 0; i < num_elements; ++i) {
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];
        
        float m_hat = m[i] / (1 - std::pow(beta1, timestep));
        float v_hat = v[i] / (1 - std::pow(beta2, timestep));
        
        params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

