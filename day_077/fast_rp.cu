#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_CURAND_ERROR(val) check_curand((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

void check_curand(curandStatus_t result, const char* func, const char* file, int line) {
    if (result != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "CURAND error at %s:%d code=%d \"%s\"\n",
                file, line, (int)result, func);
        exit(EXIT_FAILURE);
    }
}

// CPU implementation of random projection
void random_projection_cpu(const float* data, int n_samples, int n_features, 
                          int n_components, float* projected) {
    // Generate random matrix (Gaussian distribution)
    float* random_matrix = (float*)malloc(n_features * n_components * sizeof(float));
    
    srand(time(NULL));
    for (int i = 0; i < n_features * n_components; i++) {
        // Box-Muller transform for Gaussian random numbers
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        random_matrix[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }

    // Scale by 1/sqrt(n_components) for Johnson-Lindenstrauss lemma
    float scale = 1.0f / sqrtf(n_components);
    for (int i = 0; i < n_features * n_components; i++) {
        random_matrix[i] *= scale;
    }

    // Perform projection: projected = data * random_matrix
    for (int s = 0; s < n_samples; s++) {
        for (int c = 0; c < n_components; c++) {
            float sum = 0.0f;
            for (int f = 0; f < n_features; f++) {
                sum += data[s * n_features + f] * random_matrix[f * n_components + c];
            }
            projected[s * n_components + c] = sum;
        }
    }

    free(random_matrix);
}

// GPU kernel to initialize random number generators
__global__ void init_curand_kernel(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// GPU kernel to generate random matrix with Gaussian distribution
__global__ void generate_random_matrix_kernel(curandState* state, float* random_matrix, 
                                            int n_features, int n_components) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features * n_components) return;

    // Generate Gaussian random numbers
    float r = curand_normal(&state[idx]);
    random_matrix[idx] = r;
}

// GPU kernel for random projection
__global__ void project_data_kernel(const float* data, const float* random_matrix,
                                   int n_samples, int n_features, int n_components,
                                   float* projected) {
    int s = blockIdx.y * blockDim.y + threadIdx.y; // sample index
    int c = blockIdx.x * blockDim.x + threadIdx.x; // component index

    if (s >= n_samples || c >= n_components) return;

    float sum = 0.0f;
    for (int f = 0; f < n_features; f++) {
        sum += data[s * n_features + f] * random_matrix[f * n_components + c];
    }
    projected[s * n_components + c] = sum;
}

// GPU implementation of random projection
void random_projection_gpu(const float* h_data, int n_samples, int n_features,
                          int n_components, float* h_projected) {
    float *d_data, *d_random_matrix, *d_projected;
    curandState *d_states;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, n_samples * n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_random_matrix, n_features * n_components * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_projected, n_samples * n_components * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_states, n_features * n_components * sizeof(curandState)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize random number generators
    dim3 blockDim(256);
    dim3 gridDim((n_features * n_components + blockDim.x - 1) / blockDim.x);
    init_curand_kernel<<<gridDim, blockDim>>>(d_states, time(NULL));
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Generate random matrix
    generate_random_matrix_kernel<<<gridDim, blockDim>>>(d_states, d_random_matrix, n_features, n_components);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Scale random matrix by 1/sqrt(n_components)
    float scale = 1.0f / sqrtf(n_components);
    CHECK_CUDA_ERROR(cublasSscal(cublas_handle, n_features * n_components, &scale, d_random_matrix, 1));

    // Perform projection
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((n_components + blockDim2D.x - 1) / blockDim2D.x,
                  (n_samples + blockDim2D.y - 1) / blockDim2D.y);
    project_data_kernel<<<gridDim2D, blockDim2D>>>(d_data, d_random_matrix, 
                                                 n_samples, n_features, n_components, 
                                                 d_projected);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_projected, d_projected, n_samples * n_components * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_random_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_projected));
    CHECK_CUDA_ERROR(cudaFree(d_states));
}

// Generate random data
float* generate_random_data(int n_samples, int n_features) {
    float* data = (float*)malloc(n_samples * n_features * sizeof(float));
    if (!data) return NULL;

    srand(time(NULL));
    for (int i = 0; i < n_samples * n_features; i++) {
        data[i] = (float)rand() / RAND_MAX * 10.0f;
    }
    return data;
}

// Compare two matrices with tolerance
int compare_matrices(const float* a, const float* b, int rows, int cols, float tolerance) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Since random projection is approximate, we use a more lenient tolerance
            if (fabs(a[i * cols + j] - b[i * cols + j]) > tolerance) {
                printf("Mismatch at (%d,%d): %f vs %f\n", i, j, a[i * cols + j], b[i * cols + j]);
                return 0;
            }
        }
    }
    return 1;
}

// Print the first few elements of a matrix
void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s (first 5x5):\n", name);
    int print_rows = (rows < 5) ? rows : 5;
    int print_cols = (cols < 5) ? cols : 5;
    
    for (int i = 0; i < print_rows; i++) {
        for (int j = 0; j < print_cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void run_random_projection_test(int n_samples, int n_features, int n_components) {
    printf("\nRunning Random Projection test with %d samples, %d features -> %d components\n",
           n_samples, n_features, n_components);
    
    // Generate random data
    float* data = generate_random_data(n_samples, n_features);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for test data\n");
        return;
    }

    // Allocate memory for results
    float* cpu_projected = (float*)malloc(n_samples * n_components * sizeof(float));
    float* gpu_projected = (float*)malloc(n_samples * n_components * sizeof(float));

    // Run CPU random projection
    clock_t cpu_start = clock();
    random_projection_cpu(data, n_samples, n_features, n_components, cpu_projected);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Run GPU random projection
    clock_t gpu_start = clock();
    random_projection_gpu(data, n_samples, n_features, n_components, gpu_projected);
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    // Print some results
    print_matrix(cpu_projected, n_samples, n_components, "CPU Projection");
    print_matrix(gpu_projected, n_samples, n_components, "GPU Projection");

    // Validate results (with higher tolerance due to random nature)
    float tolerance = 0.1f; // Higher tolerance for random projection
    int valid = compare_matrices(cpu_projected, gpu_projected, n_samples, n_components, tolerance);

    printf("\nResults:\n");
    printf("  CPU Time: %.4f seconds\n", cpu_time);
    printf("  GPU Time: %.4f seconds\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("  Validation: %s (tolerance=%.2f)\n", valid ? "PASSED" : "FAILED", tolerance);

    // Free memory
    free(data);
    free(cpu_projected);
    free(gpu_projected);
}

int main() {
    // Initialize cuBLAS handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));

    // Test with different configurations
    run_random_projection_test(1000, 100, 10);
    run_random_projection_test(5000, 500, 50);
    run_random_projection_test(10000, 1000, 100);
    
    // Larger test
    run_random_projection_test(50000, 2000, 200);

    // Clean up cuBLAS handle
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));

    return 0;
}