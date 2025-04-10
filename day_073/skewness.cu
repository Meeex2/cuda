#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d code=%d(%s)\n", \
                   __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// CPU reference implementation (two-pass for better accuracy)
float skewness_cpu(float* data, int n) {
    // First pass: compute mean
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    float mean = sum / n;
    
    // Second pass: compute moments
    float m2 = 0.0f, m3 = 0.0f;
    for (int i = 0; i < n; i++) {
        float dev = data[i] - mean;
        float dev2 = dev * dev;
        m2 += dev2;
        m3 += dev2 * dev;
    }
    m2 /= n;
    m3 /= n;
    
    return m3 / powf(m2, 1.5f);  // skewness = m3 / σ³
}

// GPU kernel to compute partial sums (mean, m2, m3)
__global__ void skewness_kernel(float* data, int n, float* mean, float* m2, float* m3) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local sums
    float my_sum = 0.0f;
    float my_m2 = 0.0f;
    float my_m3 = 0.0f;
    
    if (i < n) {
        float val = data[i];
        my_sum = val;
    }
    
    // Reduce within block for mean
    sdata[tid] = my_sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(mean, sdata[0]);
    }
    __syncthreads();
    
    // Compute centered moments after mean is available
    float global_mean = *mean / n;
    if (i < n) {
        float dev = data[i] - global_mean;
        float dev2 = dev * dev;
        my_m2 = dev2;
        my_m3 = dev2 * dev;
    }
    
    // Reduce within block for m2 and m3
    sdata[tid] = my_m2;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(m2, sdata[0]);
    }
    
    sdata[tid] = my_m3;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(m3, sdata[0]);
    }
}

// Wrapper function for GPU computation
float skewness_gpu(float* h_data, int n) {
    float *d_data;
    float h_mean = 0.0f, h_m2 = 0.0f, h_m3 = 0.0f;
    float *d_mean, *d_m2, *d_m3;
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mean, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m2, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m3, sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mean, &h_mean, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_m2, &h_m2, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_m3, &h_m3, sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    size_t shared_size = block_size * sizeof(float);
    
    skewness_kernel<<<grid_size, block_size, shared_size>>>(d_data, n, d_mean, d_m2, d_m3);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_m2, d_m2, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_m3, d_m3, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute final skewness
    float variance = h_m2 / n;
    float m3_centered = h_m3 / n;
    float skewness = m3_centered / powf(variance, 1.5f);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_m2));
    CHECK_CUDA(cudaFree(d_m3));
    
    return skewness;
}

void test_skewness(int n) {
    // Allocate and initialize host data
    float* h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)rand() / RAND_MAX;  // Random data [0, 1]
    }
    
    // Compute results
    float cpu_result = skewness_cpu(h_data, n);
    float gpu_result = skewness_gpu(h_data, n);
    
    // Verify results
    float diff = fabs(cpu_result - gpu_result);
    printf("Test n=%d: CPU=%.6f, GPU=%.6f, Diff=%.6f\n", 
           n, cpu_result, gpu_result, diff);
    assert(diff < 1e-5);
    
    free(h_data);
}

void benchmark_skewness(int n) {
    // Allocate and initialize host data
    float* h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU benchmark
    clock_t start = clock();
    float cpu_result = skewness_cpu(h_data, n);
    double cpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
    
    // GPU benchmark (warmup first)
    skewness_gpu(h_data, 1000); // Warmup
    cudaDeviceSynchronize();
    start = clock();
    float gpu_result = skewness_gpu(h_data, n);
    cudaDeviceSynchronize();
    double gpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
    
    printf("Size n=%d: CPU %.2fms (%.2f), GPU %.2fms (%.2f), Speedup %.1fx\n",
           n, cpu_time, cpu_result, gpu_time, gpu_result, cpu_time/gpu_time);
    
    free(h_data);
}

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    // Test cases
    test_skewness(1000);
    test_skewness(10000);
    
    // Benchmark cases
    benchmark_skewness(1000000);
    benchmark_skewness(10000000);
    
    return 0;
}