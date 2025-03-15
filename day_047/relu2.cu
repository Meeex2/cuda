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

int main() {
    const int size = 1 << 20;  // 1 million elements
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float* h_data = new float[size];
    float* h_result_cpu = new float[size];
    float* h_result_gpu = new float[size];

    // Initialize data with random values
    initializeData(h_data, size);
    std::memcpy(h_result_cpu, h_data, bytes);  // Copy data for CPU computation
    std::memcpy(h_result_gpu, h_data, bytes);  // Copy data for GPU computation

    // CPU Computation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    relu2_cpu(h_result_cpu, size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Execution Time: " << cpu_time.count() << " ms\n";

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, bytes);

    // Copy data to device
    cudaMemcpy(d_data, h_result_gpu, bytes, cudaMemcpyHostToDevice);

    // CUDA Kernel Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // GPU Computation
    auto start_gpu = std::chrono::high_resolution_clock::now();
    relu2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();  // Wait for GPU to finish
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Execution Time: " << gpu_time.count() << " ms\n";

    // Copy result back to host
    cudaMemcpy(h_result_gpu, d_data, bytes, cudaMemcpyDeviceToHost);

    // Validate results
    if (compareArrays(h_result_cpu, h_result_gpu, size)) {
        std::cout << "Results match! CUDA implementation is correct.\n";
    } else {
        std::cerr << "Results do not match! CUDA implementation has errors.\n";
    }

    // Free memory
    delete[] h_data;
    delete[] h_result_cpu;
    delete[] h_result_gpu;
    cudaFree(d_data);

    return 0;
}