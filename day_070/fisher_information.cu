#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <assert.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d code=%d(%s)\n", \
                   __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// CPU reference implementation
void fisher_cpu(float* log_probs, float* fisher, int n_samples, int n_params) {
    for (int i = 0; i < n_params; i++) {
        for (int j = 0; j < n_params; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n_samples; k++) {
                sum += log_probs[k * n_params + i] * log_probs[k * n_params + j];
            }
            fisher[i * n_params + j] = sum / n_samples;
        }
    }
}

// CUDA kernel for Fisher Information Matrix
__global__ void fisher_kernel(float* log_probs, float* fisher, 
                             int n_samples, int n_params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_params && j < n_params) {
        float sum = 0.0f;
        for (int k = 0; k < n_samples; k++) {
            sum += log_probs[k * n_params + i] * log_probs[k * n_params + j];
        }
        fisher[i * n_params + j] = sum / n_samples;
    }
}

// Wrapper function for GPU computation
void fisher_gpu(float* h_log_probs, float* h_fisher, 
               int n_samples, int n_params) {
    float *d_log_probs, *d_fisher;
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_log_probs, n_samples * n_params * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fisher, n_params * n_params * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_log_probs, h_log_probs, 
                        n_samples * n_params * sizeof(float),
                        cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((n_params + block.x - 1) / block.x,
              (n_params + block.y - 1) / block.y);
    
    fisher_kernel<<<grid, block>>>(d_log_probs, d_fisher, n_samples, n_params);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_fisher, d_fisher, 
                        n_params * n_params * sizeof(float),
                        cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_log_probs));
    CHECK_CUDA(cudaFree(d_fisher));
}

// Test function
void test_fisher(int n_samples, int n_params) {
    // Allocate host memory
    float* h_log_probs = (float*)malloc(n_samples * n_params * sizeof(float));
    float* h_fisher_cpu = (float*)malloc(n_params * n_params * sizeof(float));
    float* h_fisher_gpu = (float*)malloc(n_params * n_params * sizeof(float));
    
    // Initialize random data
    for (int i = 0; i < n_samples * n_params; i++) {
        h_log_probs[i] = (float)rand() / RAND_MAX;
    }
    
    // Compute on CPU
    fisher_cpu(h_log_probs, h_fisher_cpu, n_samples, n_params);
    
    // Compute on GPU
    fisher_gpu(h_log_probs, h_fisher_gpu, n_samples, n_params);
    
    // Verify results
    float max_diff = 0.0f;
    for (int i = 0; i < n_params * n_params; i++) {
        float diff = fabs(h_fisher_cpu[i] - h_fisher_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("Test %dx%d: Max difference = %.6f\n", 
           n_samples, n_params, max_diff);
    assert(max_diff < 1e-5);
    
    // Cleanup
    free(h_log_probs);
    free(h_fisher_cpu);
    free(h_fisher_gpu);
}

// Benchmark function
void benchmark_fisher(int n_samples, int n_params) {
    // Allocate host memory
    float* h_log_probs = (float*)malloc(n_samples * n_params * sizeof(float));
    float* h_fisher = (float*)malloc(n_params * n_params * sizeof(float));
    
    // Initialize random data
    for (int i = 0; i < n_samples * n_params; i++) {
        h_log_probs[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU benchmark
    clock_t start = clock();
    fisher_cpu(h_log_probs, h_fisher, n_samples, n_params);
    double cpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
    
    // GPU benchmark
    start = clock();
    fisher_gpu(h_log_probs, h_fisher, n_samples, n_params);
    cudaDeviceSynchronize();
    double gpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
    
    printf("Size %dx%d: CPU %.2fms, GPU %.2fms, Speedup %.1fx\n",
           n_samples, n_params, cpu_time, gpu_time, cpu_time/gpu_time);
    
    // Cleanup
    free(h_log_probs);
    free(h_fisher);
}

int main() {
    // Test cases
    test_fisher(1000, 64);
    test_fisher(10000, 128);
    
    // Benchmark cases
    benchmark_fisher(1000, 64);
    benchmark_fisher(10000, 128);
    benchmark_fisher(100000, 256);
    
    return 0;
}