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

// CPU implementation
void naive_bayes_cpu(float* features, int* labels, float* class_probs, 
                    float* feature_probs, int n_samples, int n_features, int n_classes) {
    // Calculate class probabilities
    for (int c = 0; c < n_classes; c++) {
        int count = 0;
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] == c) count++;
        }
        class_probs[c] = (float)count / n_samples;
    }

    // Calculate feature probabilities (Gaussian naive Bayes)
    for (int c = 0; c < n_classes; c++) {
        for (int f = 0; f < n_features; f++) {
            float sum = 0.0f, sum_sq = 0.0f;
            int count = 0;
            
            for (int i = 0; i < n_samples; i++) {
                if (labels[i] == c) {
                    float val = features[i * n_features + f];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
            
            float mean = sum / count;
            float var = (sum_sq / count) - (mean * mean);
            feature_probs[c * n_features * 2 + f * 2] = mean;
            feature_probs[c * n_features * 2 + f * 2 + 1] = var;
        }
    }
}

// GPU kernel for calculating class probabilities
__global__ void nb_class_probs_kernel(int* labels, float* class_probs, int n_samples, int n_classes) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_classes) return;

    int count = 0;
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == c) count++;
    }
    class_probs[c] = (float)count / n_samples;
}

// GPU kernel for calculating feature probabilities
__global__ void nb_feature_probs_kernel(float* features, int* labels, float* feature_probs, 
                                       int n_samples, int n_features, int n_classes) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= n_classes || f >= n_features) return;

    float sum = 0.0f, sum_sq = 0.0f;
    int count = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == c) {
            float val = features[i * n_features + f];
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    
    float mean = sum / count;
    float var = (sum_sq / count) - (mean * mean);
    
    feature_probs[c * n_features * 2 + f * 2] = mean;
    feature_probs[c * n_features * 2 + f * 2 + 1] = var;
}

// GPU wrapper function
void naive_bayes_gpu(float* h_features, int* h_labels, float* h_class_probs, 
                    float* h_feature_probs, int n_samples, int n_features, int n_classes) {
    float *d_features, *d_class_probs, *d_feature_probs;
    int *d_labels;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_features, n_samples * n_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_labels, n_samples * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_class_probs, n_classes * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_feature_probs, n_classes * n_features * 2 * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_features, h_features, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, h_labels, n_samples * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate class probabilities
    dim3 block_class(256);
    dim3 grid_class((n_classes + block_class.x - 1) / block_class.x);
    nb_class_probs_kernel<<<grid_class, block_class>>>(d_labels, d_class_probs, n_samples, n_classes);
    CHECK_CUDA(cudaGetLastError());

    // Calculate feature probabilities
    dim3 block_feature(16, 16);
    dim3 grid_feature((n_features + block_feature.x - 1) / block_feature.x,
                     (n_classes + block_feature.y - 1) / block_feature.y);
    nb_feature_probs_kernel<<<grid_feature, block_feature>>>(d_features, d_labels, d_feature_probs, 
                                                            n_samples, n_features, n_classes);
    CHECK_CUDA(cudaGetLastError());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_class_probs, d_class_probs, n_classes * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_feature_probs, d_feature_probs, n_classes * n_features * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA(cudaFree(d_features));
    CHECK_CUDA(cudaFree(d_labels));
    CHECK_CUDA(cudaFree(d_class_probs));
    CHECK_CUDA(cudaFree(d_feature_probs));
}

// Test function
void test_naive_bayes(int n_samples, int n_features, int n_classes) {
    // Generate random data
    float* h_features = (float*)malloc(n_samples * n_features * sizeof(float));
    int* h_labels = (int*)malloc(n_samples * sizeof(int));
    float* h_class_probs_cpu = (float*)malloc(n_classes * sizeof(float));
    float* h_class_probs_gpu = (float*)malloc(n_classes * sizeof(float));
    float* h_feature_probs_cpu = (float*)malloc(n_classes * n_features * 2 * sizeof(float));
    float* h_feature_probs_gpu = (float*)malloc(n_classes * n_features * 2 * sizeof(float));

    for (int i = 0; i < n_samples; i++) {
        h_labels[i] = rand() % n_classes;
        for (int j = 0; j < n_features; j++) {
            h_features[i * n_features + j] = (float)rand() / RAND_MAX;
        }
    }

    // Compute on CPU
    naive_bayes_cpu(h_features, h_labels, h_class_probs_cpu, h_feature_probs_cpu, 
                   n_samples, n_features, n_classes);

    // Compute on GPU
    naive_bayes_gpu(h_features, h_labels, h_class_probs_gpu, h_feature_probs_gpu, 
                   n_samples, n_features, n_classes);

    // Verify results
    float max_diff = 0.0f;
    for (int i = 0; i < n_classes * n_features * 2; i++) {
        float diff = fabs(h_feature_probs_cpu[i] - h_feature_probs_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Test %dx%dx%d: Max difference = %.6f\n", 
           n_samples, n_features, n_classes, max_diff);
    assert(max_diff < 1e-5);

    free(h_features);
    free(h_labels);
    free(h_class_probs_cpu);
    free(h_class_probs_gpu);
    free(h_feature_probs_cpu);
    free(h_feature_probs_gpu);
}

// Benchmark function
void benchmark_naive_bayes(int n_samples, int n_features, int n_classes) {
    // Allocate memory
    float* h_features = (float*)malloc(n_samples * n_features * sizeof(float));
    int* h_labels = (int*)malloc(n_samples * sizeof(int));
    float* h_class_probs = (float*)malloc(n_classes * sizeof(float));
    float* h_feature_probs = (float*)malloc(n_classes * n_features * 2 * sizeof(float));

    // Initialize data
    for (int i = 0; i < n_samples; i++) {
        h_labels[i] = rand() % n_classes;
        for (int j = 0; j < n_features; j++) {
            h_features[i * n_features + j] = (float)rand() / RAND_MAX;
        }
    }

    // CPU benchmark
    clock_t start = clock();
    naive_bayes_cpu(h_features, h_labels, h_class_probs, h_feature_probs, 
                   n_samples, n_features, n_classes);
    double cpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;

    // GPU benchmark
    start = clock();
    naive_bayes_gpu(h_features, h_labels, h_class_probs, h_feature_probs, 
                   n_samples, n_features, n_classes);
    cudaDeviceSynchronize();
    double gpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;

    printf("Size %dx%dx%d: CPU %.2fms, GPU %.2fms, Speedup %.1fx\n",
           n_samples, n_features, n_classes, cpu_time, gpu_time, cpu_time/gpu_time);

    free(h_features);
    free(h_labels);
    free(h_class_probs);
    free(h_feature_probs);
}

int main() {
    // Set random seed
    srand(42);

    // Test cases
    test_naive_bayes(1000, 10, 2);
    test_naive_bayes(10000, 20, 3);

    // Benchmark cases
    benchmark_naive_bayes(100000, 50, 5);
    benchmark_naive_bayes(1000000, 100, 10);

    return 0;
}