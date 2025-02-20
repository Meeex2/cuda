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

bool validate_arrays(const float* cpu_arr, const float* gpu_arr, int num_elements, float tolerance = 1e-5) {
    for (int i = 0; i < num_elements; ++i) {
        if (std::fabs(cpu_arr[i] - gpu_arr[i]) > tolerance) {
            std::cout << "Mismatch at index " << i 
                      << ": CPU=" << cpu_arr[i] 
                      << ", GPU=" << gpu_arr[i] 
                      << std::endl;
            return false;
        }
    }
    return true;
}
int main() {
    const int num_elements = 1 << 20;  
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    const int timestep = 1;
    
    const float learning_rate = 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> host_params(num_elements);
    std::vector<float> host_grads(num_elements);
    std::vector<float> host_m(num_elements, 0.0f);
    std::vector<float> host_v(num_elements, 0.0f);
    for (int i = 0; i < num_elements; ++i) {
        host_params[i] = dist(gen);
        host_grads[i] = dist(gen);
    }
    
    float *d_params, *d_grads, *d_m, *d_v;
    cudaMalloc(&d_params, num_elements * sizeof(float));
    cudaMalloc(&d_grads, num_elements * sizeof(float));
    cudaMalloc(&d_m, num_elements * sizeof(float));
    cudaMalloc(&d_v, num_elements * sizeof(float));
    
    cudaMemcpy(d_params, host_params.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads, host_grads.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, host_m.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, host_v.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    std::vector<float> cpu_params = host_params;
    std::vector<float> cpu_m = host_m;
    std::vector<float> cpu_v = host_v;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_adam(cpu_params.data(), host_grads.data(), cpu_m.data(), cpu_v.data(),
            learning_rate, beta1, beta2, epsilon, timestep, num_elements);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    adam_kernel<<<grid_size, block_size>>>(d_params, d_grads, d_m, d_v,
                                         learning_rate, beta1, beta2, epsilon,
                                         timestep, num_elements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  
    
    std::vector<float> gpu_params(num_elements);
    std::vector<float> gpu_m(num_elements);
    std::vector<float> gpu_v(num_elements);
    
    cudaMemcpy(gpu_params.data(), d_params, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_m.data(), d_m, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_v.data(), d_v, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool params_valid = validate_arrays(cpu_params.data(), gpu_params.data(), num_elements);
    bool m_valid = validate_arrays(cpu_m.data(), gpu_m.data(), num_elements);
    bool v_valid = validate_arrays(cpu_v.data(), gpu_v.data(), num_elements);
    
    std::cout << "Validation results:\n";
    std::cout << "Parameters: " << (params_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "First moment (m): " << (m_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "Second moment (v): " << (v_valid ? "PASSED" : "FAILED") << "\n";
    
    std::cout << "\nPerformance:\n";
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";
    
    cudaFree(d_params);
    cudaFree(d_grads);
    cudaFree(d_m);
    cudaFree(d_v);
    return 0;
}
