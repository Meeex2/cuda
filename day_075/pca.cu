#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)
#define CHECK_CUSOLVER_ERROR(val) check_cusolver((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

void check_cublas(cublasStatus_t result, const char* func, const char* file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error at %s:%d code=%d \"%s\"\n",
                file, line, (int)result, func);
        exit(EXIT_FAILURE);
    }
}

void check_cusolver(cusolverStatus_t result, const char* func, const char* file, int line) {
    if (result != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUSOLVER error at %s:%d code=%d \"%s\"\n",
                file, line, (int)result, func);
        exit(EXIT_FAILURE);
    }
}

// CPU implementation of PCA
void pca_cpu(const float* data, int n_samples, int n_features, int n_components, 
             float* components, float* explained_variance) {
    // Allocate memory
    float* mean = (float*)malloc(n_features * sizeof(float));
    float* cov = (float*)malloc(n_features * n_features * sizeof(float));
    float* centered_data = (float*)malloc(n_samples * n_features * sizeof(float));

    // 1. Compute mean
    for (int f = 0; f < n_features; f++) {
        mean[f] = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            mean[f] += data[s * n_features + f];
        }
        mean[f] /= n_samples;
    }

    // 2. Center the data
    for (int s = 0; s < n_samples; s++) {
        for (int f = 0; f < n_features; f++) {
            centered_data[s * n_features + f] = data[s * n_features + f] - mean[f];
        }
    }

    // 3. Compute covariance matrix (column-major)
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_features; j++) {
            cov[j * n_features + i] = 0.0f; // Note: column-major
            for (int s = 0; s < n_samples; s++) {
                cov[j * n_features + i] += centered_data[s * n_features + i] * centered_data[s * n_features + j];
            }
            cov[j * n_features + i] /= (n_samples - 1);
        }
    }

    // 4. Compute eigenvalues and eigenvectors (using simple power iteration)
    float* eigenvectors = (float*)malloc(n_features * n_features * sizeof(float));
    float* eigenvalues = (float*)malloc(n_features * sizeof(float));

    // Initialize eigenvectors to identity
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_features; j++) {
            eigenvectors[i * n_features + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Simple power iteration (for demonstration - not optimal)
    for (int comp = 0; comp < n_components; comp++) {
        float* vec = &eigenvectors[comp * n_features];
        float eigenvalue = 0.0f;
        
        // Power iteration
        for (int iter = 0; iter < 100; iter++) {
            float* new_vec = (float*)malloc(n_features * sizeof(float));
            
            // Multiply covariance matrix with vector
            for (int i = 0; i < n_features; i++) {
                new_vec[i] = 0.0f;
                for (int j = 0; j < n_features; j++) {
                    new_vec[i] += cov[i * n_features + j] * vec[j];
                }
            }
            
            // Normalize
            float norm = 0.0f;
            for (int i = 0; i < n_features; i++) {
                norm += new_vec[i] * new_vec[i];
            }
            norm = sqrtf(norm);
            
            for (int i = 0; i < n_features; i++) {
                vec[i] = new_vec[i] / norm;
            }
            
            free(new_vec);
        }
        
        // Compute eigenvalue
        float* temp = (float*)malloc(n_features * sizeof(float));
        for (int i = 0; i < n_features; i++) {
            temp[i] = 0.0f;
            for (int j = 0; j < n_features; j++) {
                temp[i] += cov[i * n_features + j] * vec[j];
            }
        }
        
        eigenvalue = 0.0f;
        for (int i = 0; i < n_features; i++) {
            eigenvalue += vec[i] * temp[i];
        }
        eigenvalues[comp] = eigenvalue;
        
        free(temp);
        
        // Deflate the matrix
        for (int i = 0; i < n_features; i++) {
            for (int j = 0; j < n_features; j++) {
                cov[i * n_features + j] -= eigenvalue * vec[i] * vec[j];
            }
        }
    }

    // Copy results
    for (int i = 0; i < n_components; i++) {
        explained_variance[i] = eigenvalues[i];
        for (int j = 0; j < n_features; j++) {
            components[i * n_features + j] = eigenvectors[i * n_features + j];
        }
    }

    // Free memory
    free(mean);
    free(cov);
    free(centered_data);
    free(eigenvectors);
    free(eigenvalues);
}

// GPU implementation of PCA using cuBLAS and cuSOLVER
void pca_gpu(const float* h_data, int n_samples, int n_features, int n_components,
             float* h_components, float* h_explained_variance) {
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolver_handle));

    // Allocate device memory
    float *d_data, *d_centered, *d_mean, *d_cov;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, n_samples * n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_centered, n_samples * n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_mean, n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cov, n_features * n_features * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice));

    // 1. Compute mean
    const float alpha = 1.0f / n_samples;
    for (int f = 0; f < n_features; f++) {
        CHECK_CUBLAS_ERROR(cublasSasum(cublas_handle, n_samples, &d_data[f * n_samples], 1, &d_mean[f]));
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_mean, d_mean, n_features * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUBLAS_ERROR(cublasSscal(cublas_handle, n_features, &alpha, d_mean, 1));

    // 2. Center the data
    // Subtract mean from each feature (column)
    for (int f = 0; f < n_features; f++) {
        CHECK_CUBLAS_ERROR(cublasScopy(cublas_handle, n_samples, &d_data[f * n_samples], 1, &d_centered[f * n_samples], 1));
        CHECK_CUBLAS_ERROR(cublasSaxpy(cublas_handle, n_samples, &alpha, d_mean, 0, &d_centered[f * n_samples], 1));
    }

    // 3. Compute covariance matrix (1/(n-1) * X^T * X)
    const float beta = 0.0f;
    const float scale = 1.0f / (n_samples - 1);
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 n_features, n_features, n_samples,
                                 &scale, d_centered, n_samples,
                                 d_centered, n_samples,
                                 &beta, d_cov, n_features));

    // 4. Compute eigenvalues and eigenvectors using cuSOLVER
    float *d_eigenvalues, *d_eigenvectors;
    int *d_info;
    float *d_work;
    int lwork = 0;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_eigenvalues, n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_eigenvectors, n_features * n_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_info, sizeof(int)));

    // Query workspace size
    CHECK_CUSOLVER_ERROR(cusolverDnSsyevd_bufferSize(
        cusolver_handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n_features,
        d_cov,
        n_features,
        d_eigenvalues,
        &lwork));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work, lwork * sizeof(float)));

    // Compute eigenvalues and eigenvectors
    CHECK_CUSOLVER_ERROR(cusolverDnSsyevd(
        cusolver_handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n_features,
        d_cov,
        n_features,
        d_eigenvalues,
        d_work,
        lwork,
        d_info));

    // Check if syevd was successful
    int info;
    CHECK_CUDA_ERROR(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        fprintf(stderr, "Error: syevd returned %d\n", info);
        exit(EXIT_FAILURE);
    }

    // The eigenvectors are stored in d_cov in column-major order
    // The eigenvalues are in ascending order, so we need to reverse them

    // Copy the top n_components eigenvectors (last columns) and eigenvalues
    float* h_eigenvalues = (float*)malloc(n_features * sizeof(float));
    float* h_eigenvectors = (float*)malloc(n_features * n_features * sizeof(float));

    CHECK_CUDA_ERROR(cudaMemcpy(h_eigenvalues, d_eigenvalues, n_features * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_eigenvectors, d_cov, n_features * n_features * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy results in descending order of eigenvalues
    for (int i = 0; i < n_components; i++) {
        h_explained_variance[i] = h_eigenvalues[n_features - 1 - i];
        for (int j = 0; j < n_features; j++) {
            h_components[i * n_features + j] = h_eigenvectors[(n_features - 1 - i) * n_features + j];
        }
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_centered));
    CHECK_CUDA_ERROR(cudaFree(d_mean));
    CHECK_CUDA_ERROR(cudaFree(d_cov));
    CHECK_CUDA_ERROR(cudaFree(d_eigenvalues));
    CHECK_CUDA_ERROR(cudaFree(d_eigenvectors));
    CHECK_CUDA_ERROR(cudaFree(d_info));
    CHECK_CUDA_ERROR(cudaFree(d_work));

    // Free host memory
    free(h_eigenvalues);
    free(h_eigenvectors);

    // Destroy handles
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
    CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolver_handle));
}

// Generate random data with some covariance
float* generate_random_data(int n_samples, int n_features) {
    float* data = (float*)malloc(n_samples * n_features * sizeof(float));
    if (!data) return NULL;

    srand(time(NULL));
    
    // Create correlated features
    for (int s = 0; s < n_samples; s++) {
        // Base random values
        float base1 = (float)rand() / RAND_MAX * 10.0f;
        float base2 = (float)rand() / RAND_MAX * 10.0f;
        
        for (int f = 0; f < n_features; f++) {
            if (f % 2 == 0) {
                data[s * n_features + f] = base1 + (float)rand() / RAND_MAX * 2.0f - 1.0f;
            } else {
                data[s * n_features + f] = base2 + (float)rand() / RAND_MAX * 2.0f - 1.0f;
            }
        }
    }
    
    return data;
}

// Compare two matrices with tolerance
int compare_matrices(const float* a, const float* b, int rows, int cols, float tolerance) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(a[i * cols + j] - b[i * cols + j]) > tolerance) {
                // Check for sign flip (eigenvectors can have opposite signs)
                if (fabs(a[i * cols + j] + b[i * cols + j]) > tolerance) {
                    printf("Mismatch at (%d,%d): %f vs %f\n", i, j, a[i * cols + j], b[i * cols + j]);
                    return 0;
                }
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
            printf("%8.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void run_pca_test(int n_samples, int n_features, int n_components) {
    printf("\nRunning PCA test with %d samples, %d features, %d components\n",
           n_samples, n_features, n_components);
    
    // Generate random data
    float* data = generate_random_data(n_samples, n_features);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for test data\n");
        return;
    }

    // Allocate memory for results
    float* cpu_components = (float*)malloc(n_components * n_features * sizeof(float));
    float* cpu_variance = (float*)malloc(n_components * sizeof(float));
    float* gpu_components = (float*)malloc(n_components * n_features * sizeof(float));
    float* gpu_variance = (float*)malloc(n_components * sizeof(float));

    // Run CPU PCA
    clock_t cpu_start = clock();
    pca_cpu(data, n_samples, n_features, n_components, cpu_components, cpu_variance);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Run GPU PCA
    clock_t gpu_start = clock();
    pca_gpu(data, n_samples, n_features, n_components, gpu_components, gpu_variance);
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    // Print some results
    print_matrix(cpu_components, n_components, n_features, "CPU Components");
    print_matrix(gpu_components, n_components, n_features, "GPU Components");
    
    printf("CPU Explained Variance (first 5): ");
    for (int i = 0; i < (n_components < 5 ? n_components : 5); i++) {
        printf("%8.2f ", cpu_variance[i]);
    }
    printf("\n");
    
    printf("GPU Explained Variance (first 5): ");
    for (int i = 0; i < (n_components < 5 ? n_components : 5); i++) {
        printf("%8.2f ", gpu_variance[i]);
    }
    printf("\n");

    // Validate results
    float tolerance = 1e-3f;
    int components_valid = compare_matrices(cpu_components, gpu_components, 
                                          n_components, n_features, tolerance);
    
    int variance_valid = 1;
    for (int i = 0; i < n_components; i++) {
        if (fabs(cpu_variance[i] - gpu_variance[i]) > tolerance) {
            variance_valid = 0;
            break;
        }
    }

    printf("\nResults:\n");
    printf("  CPU Time: %f seconds\n", cpu_time);
    printf("  GPU Time: %f seconds\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("  Components Validation: %s\n", components_valid ? "PASSED" : "FAILED");
    printf("  Variance Validation: %s\n", variance_valid ? "PASSED" : "FAILED");

    // Free memory
    free(data);
    free(cpu_components);
    free(cpu_variance);
    free(gpu_components);
    free(gpu_variance);
}

int main() {
    // Test with different configurations
    run_pca_test(1000, 10, 2);
    run_pca_test(5000, 20, 5);
    run_pca_test(10000, 50, 10);
    
    // Larger test 
    run_pca_test(50000, 100, 20);

    return 0;
}