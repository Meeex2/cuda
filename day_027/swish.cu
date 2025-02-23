#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

// CUDA kernel for Swish activation
__global__ void swish_kernel(const float* input, float* output, int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float x = input[i];
        output[i] = x / (1.0f + __expf(-x));  // x * sigmoid(x)
    }
}

// CPU reference implementation
void cpu_swish(const float* input, float* output, int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        const float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

// Validation function
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
    const int num_elements = 1 << 24;  // 16.7M elements
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    // Initialize host data with fixed seed
    std::vector<float> host_input(num_elements);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    for (int i = 0; i < num_elements; ++i) {
        host_input[i] = dist(gen);
    }

    // Allocate device memory
    float *device_input, *device_output;
    cudaMalloc(&device_input, num_elements * sizeof(float));
    cudaMalloc(&device_output, num_elements * sizeof(float));

    // Copy input to device
    cudaMemcpy(device_input, host_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Run CPU version
    std::vector<float> cpu_output(num_elements);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_swish(host_input.data(), cpu_output.data(), num_elements);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    // Run GPU version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    swish_kernel<<<grid_size, block_size>>>(device_input, device_output, num_elements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  // Convert to seconds

    // Copy GPU results back
    std::vector<float> gpu_output(num_elements);
    cudaMemcpy(gpu_output.data(), device_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate results
    bool validation = validate_results(cpu_output.data(), gpu_output.data(), num_elements);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;

    // Print timings
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    // Cleanup
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}