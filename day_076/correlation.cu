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

// CPU implementation of correlation matrix
void correlation_cpu(const float* data, int n_samples, int n_features, float* corr) {
    // Calculate means
    float* means = (float*)malloc(n_features * sizeof(float));
    for (int f = 0; f < n_features; f++) {
        means[f] = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            means[f] += data[s * n_features + f];
        }
        means[f] /= n_samples;
    }

    // Calculate standard deviations
    float* stddevs = (float*)malloc(n_features * sizeof(float));
    for (int f = 0; f < n_features; f++) {
        stddevs[f] = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            float diff = data[s * n_features + f] - means[f];
            stddevs[f] += diff * diff;
        }
        stddevs[f] = sqrtf(stddevs[f] / n_samples);
    }

    // Calculate correlation matrix
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_features; j++) {
            float cov = 0.0f;
            for (int s = 0; s < n_samples; s++) {
                cov += (data[s * n_features + i] - means[i]) * 
                       (data[s * n_features + j] - means[j]);
            }
            cov /= n_samples;
            corr[i * n_features + j] = cov / (stddevs[i] * stddevs[j]);
        }
    }

    free(means);
    free(stddevs);
}

// GPU kernel for calculating means
__global__ void calculate_means_kernel(const float* data, int n_samples, int n_features, float* means) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_features) return;

    float sum = 0.0f;
    for (int s = 0; s < n_samples; s++) {
        sum += data[s * n_features + f];
    }
    means[f] = sum / n_samples;
}

// GPU kernel for calculating standard deviations
__global__ void calculate_stddevs_kernel(const float* data, const float* means, 
                                        int n_samples, int n_features, float* stddevs) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_features) return;

    float sum = 0.0f;
    float mean = means[f];
    for (int s = 0; s < n_samples; s++) {
        float diff = data[s * n_features + f] - mean;
        sum += diff * diff;
    }
    stddevs[f] = sqrtf(sum / n_samples);
}

// GPU kernel for calculating correlation matrix
__global__ void calculate_correlation_kernel(const float* data, const float* means, const float* stddevs,
                                           int n_samples, int n_features, float* corr) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n_features || j >= n_features) return;

    float cov = 0.0f;
    float mean_i = means[i];
    float mean_j = means[j];
    
    for (int s = 0; s < n_samples; s++) {
        cov += (data[s * n_features + i] - mean_i) * 
               (data[s * n_features + j] - mean_j);
    }
    cov /= n_samples;
    
    corr[i * n_features + j] = cov / (stddevs[i] * stddevs[j]);
}

// GPU implementation of correlation matrix
void correlation_gpu(const float* h_data, int n_samples, int n_features, float* h_corr) {
    float *d_data, *d_means, *d_stddevs, *d_corr;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, n_samples * n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_means, n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_stddevs, n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_corr, n_features * n_features * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate means
    dim3 blockDim(256);
    dim3 gridDim((n_features + blockDim.x - 1) / blockDim.x);
    calculate_means_kernel<<<gridDim, blockDim>>>(d_data, n_samples, n_features, d_means);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Calculate standard deviations
    calculate_stddevs_kernel<<<gridDim, blockDim>>>(d_data, d_means, n_samples, n_features, d_stddevs);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Calculate correlation matrix
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((n_features + blockDim2D.x - 1) / blockDim2D.x,
                  (n_features + blockDim2D.y - 1) / blockDim2D.y);
    calculate_correlation_kernel<<<gridDim2D, blockDim2D>>>(d_data, d_means, d_stddevs, 
                                                           n_samples, n_features, d_corr);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_corr, d_corr, n_features * n_features * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_means));
    CHECK_CUDA_ERROR(cudaFree(d_stddevs));
    CHECK_CUDA_ERROR(cudaFree(d_corr));
}

// Generate random data with some correlation
float* generate_random_data(int n_samples, int n_features) {
    float* data = (float*)malloc(n_samples * n_features * sizeof(float));
    if (!data) return NULL;

    srand(time(NULL));
    
    // Create correlated features (every 2 features are correlated)
    for (int s = 0; s < n_samples; s++) {
        float base_value = (float)rand() / RAND_MAX * 10.0f;
        for (int f = 0; f < n_features; f++) {
            if (f % 2 == 0) {
                data[s * n_features + f] = base_value + ((float)rand() / RAND_MAX - 0.5f);
            } else {
                data[s * n_features + f] = base_value * 0.5f + ((float)rand() / RAND_MAX - 0.5f);
            }
        }
    }
    
    return data;
}

// Compare two matrices with tolerance
int compare_matrices(const float* a, const float* b, int n, float tolerance) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(a[i * n + j] - b[i * n + j]) > tolerance) {
                printf("Mismatch at (%d,%d): %f vs %f\n", i, j, a[i * n + j], b[i * n + j]);
                return 0;
            }
        }
    }
    return 1;
}

// Print the first few elements of the correlation matrix
void print_correlation_matrix(const float* corr, int n_features) {
    printf("Correlation Matrix (first 5x5):\n");
    int print_size = n_features < 5 ? n_features : 5;
    
    for (int i = 0; i < print_size; i++) {
        for (int j = 0; j < print_size; j++) {
            printf("%6.3f ", corr[i * n_features + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void run_correlation_test(int n_samples, int n_features) {
    printf("\nRunning correlation test with %d samples, %d features\n", n_samples, n_features);
    
    // Generate random data
    float* data = generate_random_data(n_samples, n_features);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for test data\n");
        return;
    }

    // Allocate memory for results
    float* cpu_corr = (float*)malloc(n_features * n_features * sizeof(float));
    float* gpu_corr = (float*)malloc(n_features * n_features * sizeof(float));

    // Run CPU correlation
    clock_t cpu_start = clock();
    correlation_cpu(data, n_samples, n_features, cpu_corr);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Run GPU correlation
    clock_t gpu_start = clock();
    correlation_gpu(data, n_samples, n_features, gpu_corr);
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    // Print some results
    print_correlation_matrix(cpu_corr, n_features);
    print_correlation_matrix(gpu_corr, n_features);

    // Validate results
    float tolerance = 1e-3f;
    int valid = compare_matrices(cpu_corr, gpu_corr, n_features, tolerance);

    printf("\nResults:\n");
    printf("  CPU Time: %.4f seconds\n", cpu_time);
    printf("  GPU Time: %.4f seconds\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");

    // Free memory
    free(data);
    free(cpu_corr);
    free(gpu_corr);
}

int main() {
    // Test with different configurations
    run_correlation_test(1000, 10);
    run_correlation_test(5000, 20);
    run_correlation_test(10000, 50);
    
    // Larger test - comment out if your GPU doesn't have enough memory
    run_correlation_test(50000, 100);

    return 0;
}