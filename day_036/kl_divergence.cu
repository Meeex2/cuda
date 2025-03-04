#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// CUDA kernel for KL divergence
__global__ void kl_divergence_kernel(const float* P, const float* Q, float* divergence, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = P[idx];
        float q = Q[idx];
        divergence[idx] = p * logf(p / q);
    }
}

// CPU reference implementation of KL divergence
float kl_divergence_cpu(const std::vector<float>& P, const std::vector<float>& Q, int size) {
    float divergence = 0.0f;
    for (int i = 0; i < size; i++) {
        divergence += P[i] * logf(P[i] / Q[i]);
    }
    return divergence;
}

// Validation function
bool validate_results(float cpu_divergence, float gpu_divergence, float tolerance = 1e-5) {
    if (std::fabs(cpu_divergence - gpu_divergence) > tolerance) {
        std::cout << "Mismatch: CPU=" << cpu_divergence << ", GPU=" << gpu_divergence << std::endl;
        return false;
    }
    return true;
}

int main() {
    std::cout << "Starting program..." << std::endl;

    // Problem dimensions
    const int size = 1000000;  // Number of elements in the distributions

    // Allocate host memory
    std::vector<float> P(size);
    std::vector<float> Q(size);
    std::vector<float> divergence(size, 0.0f);

    // Initialize P and Q with random values
    std::mt19937 gen(42);  // Random number generator
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);  // Uniform distribution for values

    for (int i = 0; i < size; i++) {
        P[i] = dist(gen);
        Q[i] = dist(gen);
    }

    std::cout << "Initialized P and Q with random values." << std::endl;

    // Run CPU version
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_divergence = kl_divergence_cpu(P, Q, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    std::cout << "CPU version completed in " << cpu_duration << " seconds." << std::endl;

    // Allocate device memory
    float *d_P, *d_Q, *d_divergence;
    cudaMalloc(&d_P, size * sizeof(float));
    cudaMalloc(&d_Q, size * sizeof(float));
    cudaMalloc(&d_divergence, size * sizeof(float));

    std::cout << "Allocated device memory." << std::endl;

    // Copy data to device
    cudaMemcpy(d_P, P.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Copied data to device." << std::endl;

    // Define block and grid sizes
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Run GPU version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kl_divergence_kernel<<<grid_size, block_size>>>(d_P, d_Q, d_divergence, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  // Convert to seconds

    std::cout << "KL divergence kernel executed." << std::endl;

    // Copy GPU results back
    cudaMemcpy(divergence.data(), d_divergence, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Copied GPU results back to host." << std::endl;

    // Compute total divergence on GPU
    float gpu_divergence = 0.0f;
    for (int i = 0; i < size; i++) {
        gpu_divergence += divergence[i];
    }

    // Validate results
    bool validation = validate_results(cpu_divergence, gpu_divergence);
    std::cout << "Validation: " << (validation ? "PASSED (error < 1e-5)" : "FAILED") << std::endl;

    // Print timings
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_divergence);

    std::cout << "Program completed successfully." << std::endl;

    return 0;
}



