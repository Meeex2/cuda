#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

// Sigmoid function for CPU
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// GLU activation function on CPU
void gluCPU(const float* input, float* output, int n, int split_dim) {
    for (int i = 0; i < n; i += split_dim) {
        for (int j = 0; j < split_dim / 2; j++) {
            float a = input[i + j];
            float b = input[i + j + split_dim / 2];
            output[i / 2 + j] = a * sigmoid(b);
        }
    }
}

// CUDA kernel for GLU activation on GPU
__global__ void gluKernel(const float* input, float* output, int n, int split_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        int base = (idx / (split_dim / 2)) * split_dim;
        int offset = idx % (split_dim / 2);
        float a = input[base + offset];
        float b = input[base + offset + split_dim / 2];
        output[idx] = a * (1.0f / (1.0f + expf(-b)));
    }
}

// Function to check results with tolerance
bool checkResults(const float* cpu_result, const float* gpu_result, int n, float tolerance = 1e-5) {
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
    const int seq_len = 1024;          // Number of sequences
    const int feature_dim = 1024;      // Feature dimension (must be even)
    const int total_size = seq_len * feature_dim;  // Total input elements
    const int output_size = seq_len * (feature_dim / 2);  // Total output elements
    const int blockSize = 256;         // Threads per block for CUDA
    const int gridSize = (output_size + blockSize - 1) / blockSize;  // Number of blocks

    // Allocate host memory
    std::vector<float> h_input(total_size);      // Input array
    std::vector<float> h_output_cpu(output_size); // CPU output
    float* h_output_gpu = new float[output_size]; // GPU output

    // Initialize input with random values between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < total_size; i++) {
        h_input[i] = dis(gen);
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    // CPU computation and timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gluCPU(h_input.data(), h_output_cpu.data(), total_size, feature_dim);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

    // GPU computation and timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    gluKernel<<<gridSize, blockSize>>>(d_input, d_output, total_size, feature_dim);
    cudaDeviceSynchronize();  // Ensure GPU computation is complete
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    // Copy GPU results back to host
    cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    bool passed = checkResults(h_output_cpu.data(), h_output_gpu, output_size);

    // Print results
    std::cout << "Total input elements: " << total_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;
    std::cout << "CPU time: " << cpu_time << " microseconds" << std::endl;
    std::cout << "GPU time: " << gpu_time << " microseconds" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time / gpu_time << "x" << std::endl;
    
    if (passed) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output_gpu;

    return 0;
}