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


int main() {
    const int num_elements = 1 << 24;  
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    std::vector<float> host_input(num_elements);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    for (int i = 0; i < num_elements; ++i) {
        host_input[i] = dist(gen);
    }
    
    float *device_input, *device_output;
    cudaMalloc(&device_input, num_elements * sizeof(float));
    cudaMalloc(&device_output, num_elements * sizeof(float));
    
    cudaMemcpy(device_input, host_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    std::vector<float> cpu_output(num_elements);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_elu(host_input.data(), cpu_output.data(), num_elements);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    elu_kernel<<<grid_size, block_size>>>(device_input, device_output, num_elements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  
    
    std::vector<float> gpu_output(num_elements);
    cudaMemcpy(gpu_output.data(), device_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool validation = validate_results(cpu_output.data(), gpu_output.data(), num_elements);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";
    
    cudaFree(device_input);
    cudaFree(device_output);
    return 0;
}
