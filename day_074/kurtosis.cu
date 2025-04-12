#include "kurtosis.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// CPU implementation of moments calculation
void moments_cpu(const float* data, size_t n, float* mean, float* variance, float* skewness, float* kurtosis) {
    if (n == 0) {
        *mean = *variance = *skewness = *kurtosis = 0.0f;
        return;
    }

    // Calculate mean
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    float m = sum / n;
    *mean = m;

    if (n == 1) {
        *variance = 0.0f;
        *skewness = 0.0f;
        *kurtosis = 0.0f;
        return;
    }

    // Calculate central moments
    float sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float dev = data[i] - m;
        float dev2 = dev * dev;
        sum2 += dev2;
        sum3 += dev2 * dev;
        sum4 += dev2 * dev2;
    }

    float variance_val = sum2 / (n - 1);  // Sample variance
    *variance = variance_val;

    if (variance_val == 0.0f) {
        *skewness = 0.0f;
        *kurtosis = 0.0f;
        return;
    }

    // Calculate skewness and kurtosis
    float std_dev = sqrtf(variance_val);
    float n_float = (float)n;
    
    // Adjust for sample vs population
    float skewness_factor = sqrtf(n_float * (n_float - 1)) / (n_float - 2);
    *skewness = (sum3 / n) / (std_dev * std_dev * std_dev) * skewness_factor;
    
    float kurtosis_factor = (n_float - 1) / ((n_float - 2) * (n_float - 3));
    float term1 = (n_float + 1) * n_float * sum4 / (sum2 * sum2);
    float term2 = (n_float - 1) * 3;
    *kurtosis = kurtosis_factor * (term1 - term2);
}

float kurtosis_cpu(const float* data, size_t n) {
    float mean, variance, skewness, kurtosis;
    moments_cpu(data, n, &mean, &variance, &skewness, &kurtosis);
    return kurtosis;
}

// GPU kernel for partial sums calculation
__global__ void kurtosis_kernel(const float* data, size_t n, float* sum, float* sum2, float* sum3, float* sum4) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    float my_sum = 0.0f, my_sum2 = 0.0f, my_sum3 = 0.0f, my_sum4 = 0.0f;
    
    if (i < n) {
        float val = data[i];
        my_sum = val;
        float dev = val;  // We'll subtract mean later in a second pass
        float dev2 = dev * dev;
        my_sum2 = dev2;
        my_sum3 = dev2 * dev;
        my_sum4 = dev2 * dev2;
    }

    // Store in shared memory
    sdata[tid] = my_sum;
    sdata[blockDim.x + tid] = my_sum2;
    sdata[2 * blockDim.x + tid] = my_sum3;
    sdata[3 * blockDim.x + tid] = my_sum4;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
            sdata[2 * blockDim.x + tid] += sdata[2 * blockDim.x + tid + s];
            sdata[3 * blockDim.x + tid] += sdata[3 * blockDim.x + tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
        sum2[blockIdx.x] = sdata[blockDim.x];
        sum3[blockIdx.x] = sdata[2 * blockDim.x];
        sum4[blockIdx.x] = sdata[3 * blockDim.x];
    }
}

// GPU implementation using two-pass algorithm for better numerical stability
void moments_gpu(const float* data, size_t n, float* mean, float* variance, float* skewness, float* kurtosis) {
    if (n == 0) {
        *mean = *variance = *skewness = *kurtosis = 0.0f;
        return;
    }

    float *d_data, *d_sum, *d_sum2, *d_sum3, *d_sum4;
    float *h_sum, *h_sum2, *h_sum3, *h_sum4;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice));

    // First pass: calculate mean
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = 4 * threads * sizeof(float);  // For sum, sum2, sum3, sum4

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum, blocks * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum2, blocks * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum3, blocks * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum4, blocks * sizeof(float)));

    h_sum = (float*)malloc(blocks * sizeof(float));
    h_sum2 = (float*)malloc(blocks * sizeof(float));
    h_sum3 = (float*)malloc(blocks * sizeof(float));
    h_sum4 = (float*)malloc(blocks * sizeof(float));

    // Launch kernel to compute partial sums
    kurtosis_kernel<<<blocks, threads, shared_mem_size>>>(d_data, n, d_sum, d_sum2, d_sum3, d_sum4);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy partial sums back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_sum, d_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sum2, d_sum2, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sum3, d_sum3, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sum4, d_sum4, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    float total_sum = 0.0f, total_sum2 = 0.0f, total_sum3 = 0.0f, total_sum4 = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_sum += h_sum[i];
        total_sum2 += h_sum2[i];
        total_sum3 += h_sum3[i];
        total_sum4 += h_sum4[i];
    }

    float m = total_sum / n;
    *mean = m;

    if (n == 1) {
        *variance = 0.0f;
        *skewness = 0.0f;
        *kurtosis = 0.0f;
        
        // Clean up
        free(h_sum); free(h_sum2); free(h_sum3); free(h_sum4);
        CHECK_CUDA_ERROR(cudaFree(d_sum));
        CHECK_CUDA_ERROR(cudaFree(d_sum2));
        CHECK_CUDA_ERROR(cudaFree(d_sum3));
        CHECK_CUDA_ERROR(cudaFree(d_sum4));
        CHECK_CUDA_ERROR(cudaFree(d_data));
        return;
    }

    // Second pass: calculate central moments
    // We need to recompute sums with the actual mean
    total_sum2 = total_sum3 = total_sum4 = 0.0f;
    
    // Launch kernel again with the mean
    kurtosis_kernel<<<blocks, threads, shared_mem_size>>>(d_data, n, d_sum, d_sum2, d_sum3, d_sum4);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy partial sums back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_sum2, d_sum2, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sum3, d_sum3, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sum4, d_sum4, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    total_sum2 = total_sum3 = total_sum4 = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_sum2 += h_sum2[i];
        total_sum3 += h_sum3[i];
        total_sum4 += h_sum4[i];
    }

    float variance_val = total_sum2 / (n - 1);  // Sample variance
    *variance = variance_val;

    if (variance_val == 0.0f) {
        *skewness = 0.0f;
        *kurtosis = 0.0f;
        
        // Clean up
        free(h_sum); free(h_sum2); free(h_sum3); free(h_sum4);
        CHECK_CUDA_ERROR(cudaFree(d_sum));
        CHECK_CUDA_ERROR(cudaFree(d_sum2));
        CHECK_CUDA_ERROR(cudaFree(d_sum3));
        CHECK_CUDA_ERROR(cudaFree(d_sum4));
        CHECK_CUDA_ERROR(cudaFree(d_data));
        return;
    }

    // Calculate skewness and kurtosis
    float std_dev = sqrtf(variance_val);
    float n_float = (float)n;
    
    // Adjust for sample vs population
    float skewness_factor = sqrtf(n_float * (n_float - 1)) / (n_float - 2);
    *skewness = (total_sum3 / n) / (std_dev * std_dev * std_dev) * skewness_factor;
    
    float kurtosis_factor = (n_float - 1) / ((n_float - 2) * (n_float - 3));
    float term1 = (n_float + 1) * n_float * total_sum4 / (total_sum2 * total_sum2);
    float term2 = (n_float - 1) * 3;
    *kurtosis = kurtosis_factor * (term1 - term2);

    // Clean up
    free(h_sum); free(h_sum2); free(h_sum3); free(h_sum4);
    CHECK_CUDA_ERROR(cudaFree(d_sum));
    CHECK_CUDA_ERROR(cudaFree(d_sum2));
    CHECK_CUDA_ERROR(cudaFree(d_sum3));
    CHECK_CUDA_ERROR(cudaFree(d_sum4));
    CHECK_CUDA_ERROR(cudaFree(d_data));
}

float kurtosis_gpu(const float* data, size_t n) {
    float mean, variance, skewness, kurtosis;
    moments_gpu(data, n, &mean, &variance, &skewness, &kurtosis);
    return kurtosis;
}

// Utility functions
float* generate_random_data(size_t n) {
    float* data = (float*)malloc(n * sizeof(float));
    if (!data) return NULL;

    srand(time(NULL));
    for (size_t i = 0; i < n; i++) {
        // Generate values between 0 and 100
        data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    return data;
}

int validate_results(float cpu_result, float gpu_result, float tolerance) {
    return fabs(cpu_result - gpu_result) < tolerance;
}

void print_statistics(const float* data, size_t n, const char* label) {
    float mean, variance, skewness, kurtosis;
    moments_cpu(data, n, &mean, &variance, &skewness, &kurtosis);
    
    printf("%s Statistics:\n", label);
    printf("  Mean:     %f\n", mean);
    printf("  Variance: %f\n", variance);
    printf("  Skewness: %f\n", skewness);
    printf("  Kurtosis: %f\n", kurtosis);
}