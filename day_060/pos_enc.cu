#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>

// CUDA kernel for positional encoding on GPU
__global__ void positionalEncodingKernel(float* pe, int max_seq_len, int d_model) {
    int pos = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < max_seq_len && i < d_model/2) {
        float div_term = expf(-2.0f * i * logf(10000.0f) / d_model);
        float pos_float = static_cast<float>(pos);
        
        int even_idx = pos * d_model + 2 * i;
        int odd_idx = pos * d_model + 2 * i + 1;
        
        pe[even_idx] = sinf(pos_float * div_term);
        pe[odd_idx] = cosf(pos_float * div_term);
    }
}

// CPU implementation of positional encoding
void positionalEncodingCPU(float* pe, int max_seq_len, int d_model) {
    for (int pos = 0; pos < max_seq_len; pos++) {
        float pos_float = static_cast<float>(pos);
        for (int i = 0; i < d_model / 2; i++) {
            float div_term = expf(-2.0f * static_cast<float>(i) * logf(10000.0f) / static_cast<float>(d_model));
            pe[pos * d_model + 2 * i] = sinf(pos_float * div_term);
            pe[pos * d_model + 2 * i + 1] = cosf(pos_float * div_term);
        }
    }
}

// Function to verify results with a specified tolerance
bool checkResults(float* cpu_result, float* gpu_result, int n, float tolerance = 1e-4) {
    for (int i = 0; i < n; i++) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": CPU = " << cpu_result[i] 
                      << ", GPU = " << gpu_result[i] << ", diff = " << diff << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Parameters
    const int max_seq_len = 512;    // Maximum sequence length
    const int d_model = 256;        // Model dimension (must be even)
    const int total_size = max_seq_len * d_model;
    
    // Define 2D grid and block dimensions for GPU kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((d_model/2 + blockDim.x - 1) / blockDim.x, 
                 (max_seq_len + blockDim.y - 1) / blockDim.y);

    // Allocate host memory
    std::vector<float> h_pe_cpu(total_size, 0.0f);
    float* h_pe_gpu = new float[total_size]();

    // Allocate device memory
    float* d_pe;
    cudaMalloc(&d_pe, total_size * sizeof(float));

    // CPU computation with timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    positionalEncodingCPU(h_pe_cpu.data(), max_seq_len, d_model);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

    // GPU computation with timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    positionalEncodingKernel<<<gridDim, blockDim>>>(d_pe, max_seq_len, d_model);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    // Copy GPU results back to host
    cudaMemcpy(h_pe_gpu, d_pe, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    bool passed = checkResults(h_pe_cpu.data(), h_pe_gpu, total_size, 1e-4);

    // Output results
    std::cout << "Sequence length: " << max_seq_len << std::endl;
    std::cout << "Model dimension: " << d_model << std::endl;
    std::cout << "Total elements: " << total_size << std::endl;
    std::cout << "CPU time: " << cpu_time << " microseconds" << std::endl;
    std::cout << "GPU time: " << gpu_time << " microseconds" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time / gpu_time << "x" << std::endl;
    
    if (passed) {
        std::cout << "PAAASSEEED" << std::endl;
    } else {
        std::cout << "FAAAIILLEED" << std::endl;
    }

    // Cleanup
    cudaFree(d_pe);
    delete[] h_pe_gpu;

    return 0;
}