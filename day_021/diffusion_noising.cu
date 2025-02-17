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

bool validate_results(const float* cpu_data, const float* gpu_data, int num_elements, float tolerance = 1e-5) {
    for (int i = 0; i < num_elements; ++i) {
        if (fabs(cpu_data[i] - gpu_data[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu_data[i] << ", GPU=" << gpu_data[i] << std::endl;
            return false;
        }
    }
    return true;
}
int main() {
    const int num_elements = 1 << 20;  
    const int timestep = 0;
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> host_x(num_elements, 1.0f);  
    std::vector<float> host_noise(num_elements);
    std::vector<float> host_alpha_bar = {0.1f};  
    
    
    for (int i = 0; i < num_elements; ++i) {
        host_noise[i] = dist(gen);
    }
    
    float *device_x, *device_noise, *device_alpha_bar;
    cudaMalloc(&device_x, num_elements * sizeof(float));
    cudaMalloc(&device_noise, num_elements * sizeof(float));
    cudaMalloc(&device_alpha_bar, host_alpha_bar.size() * sizeof(float));
    
    cudaMemcpy(device_x, host_x.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_noise, host_noise.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_alpha_bar, host_alpha_bar.data(), host_alpha_bar.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    std::vector<float> cpu_result = host_x;
    auto cpu_start = clock();
    cpu_diffusion_noise(cpu_result.data(), host_noise.data(), host_alpha_bar.data(), timestep, num_elements);
    float cpu_duration = (float)(clock() - cpu_start) / CLOCKS_PER_SEC;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    diffusion_noise_kernel<<<grid_size, block_size>>>(device_x, device_noise, device_alpha_bar, timestep, num_elements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  
    
    std::vector<float> gpu_result(num_elements);
    cudaMemcpy(gpu_result.data(), device_x, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool validation = validate_results(cpu_result.data(), gpu_result.data(), num_elements);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";
    
    cudaFree(device_x);
    cudaFree(device_noise);
    cudaFree(device_alpha_bar);
    return 0;
}
