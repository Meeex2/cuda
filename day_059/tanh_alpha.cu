#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

// CUDA kernel for tanh(alpha * x)
__global__ void tanhKernel(float* input, float* output, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(alpha * input[idx]);
    }
}

// CPU implementation for comparison
void tanhCPU(float* input, float* output, float alpha, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = std::tanh(alpha * input[i]);
    }
}

// Function to check results
bool checkResults(float* cpu_result, float* gpu_result, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n; i++) {
        if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": CPU = " << cpu_result[i] 
                      << ", GPU = " << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Parameters
    const int N = 1 << 20; // 1M elements
    const float alpha = 2.0f;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate host memory
    std::vector<float> h_input(N);
    std::vector<float> h_output_cpu(N);
    float* h_output_gpu = new float[N];

    // Initialize input with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        h_input[i] = dis(gen);
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // CPU computation and timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    tanhCPU(h_input.data(), h_output_cpu.data(), alpha, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

    // GPU computation and timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    tanhKernel<<<gridSize, blockSize>>>(d_input, d_output, alpha, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    // Copy results back to host
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    bool passed = checkResults(h_output_cpu.data(), h_output_gpu, N);

    // Print results
    std::cout << "Array size: " << N << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "CPU time: " << cpu_time << " microseconds" << std::endl;
    std::cout << "GPU time: " << gpu_time << " microseconds" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time / gpu_time << "x" << std::endl;
    
    if (passed) {
        std::cout << "PAAASSEEED" << std::endl;
    } else {
        std::cout << "FAAAIILLEED" << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output_gpu;

    return 0;
}