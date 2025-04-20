%%writefile symmat.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)

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

void symm_mult_cpu(const float *A, int n, int k, float *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * A[j * k + l];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void symm_mult_kernel_naive(const float *A, int n, int k, float *C) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[i * k + l] * A[j * k + l];
        }
        C[i * n + j] = sum;
    }
}

__global__ void symm_mult_kernel_optimized(const float *A, int n, int k, float *C) {
    __shared__ float tile_a[32][32];
    __shared__ float tile_b[32][32];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;

    for (int m = 0; m < (k + 31)/32; m++) {
        int load_col_a = m * 32 + tx;
        if (row < n && load_col_a < k) {
            tile_a[ty][tx] = A[row * k + load_col_a];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        int load_col_b = m * 32 + ty;
        if (col < n && load_col_b < k) {
            tile_b[ty][tx] = A[col * k + load_col_b];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int l = 0; l < 32; l++) {
            sum += tile_a[ty][l] * tile_b[tx][l];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

void symm_mult_gpu_cublas(const float *h_A, int n, int k, float *h_C, cublasHandle_t handle) {
    float *d_A, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, n * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, n * n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, n * k * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Corrected cuBLAS call:
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, 
                                 CUBLAS_OP_T,  // Transpose first matrix (A^T)
                                 CUBLAS_OP_N,  // No transpose for second matrix
                                 n,            // Number of rows of result
                                 n,            // Number of columns of result
                                 k,            // Common dimension
                                 &alpha, 
                                 d_A,          // A matrix
                                 k,            // Leading dimension of A (original columns)
                                 d_A,          // B matrix (same as A)
                                 k,            // Leading dimension of B
                                 &beta, 
                                 d_C,          // C matrix
                                 n));           // Leading dimension of C

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

void generate_matrix(float *A, int n, int k) {
    srand(time(NULL));
    for (int i = 0; i < n * k; i++) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

int compare_matrices(const float *A, const float *B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at %d: %.6f vs %.6f\n", i, A[i], B[i]);
            return 0;
        }
    }
    return 1;
}

void run_test(int n, int k) {
    printf("\nTesting (%d x %d) * (%d x %d)^T\n", n, k, n, k);
    
    float *h_A = (float*)malloc(n * k * sizeof(float));
    float *h_C_cpu = (float*)malloc(n * n * sizeof(float));
    float *h_C_naive = (float*)malloc(n * n * sizeof(float));
    float *h_C_optim = (float*)malloc(n * n * sizeof(float));
    float *h_C_cublas = (float*)malloc(n * n * sizeof(float));

    generate_matrix(h_A, n, k);

    // CPU reference
    clock_t start = clock();
    symm_mult_cpu(h_A, n, k, h_C_cpu);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // GPU implementations
    float *d_A, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, n * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, n * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, n * k * sizeof(float), cudaMemcpyHostToDevice));

    // Naive kernel
    dim3 block_naive(16, 16);
    dim3 grid_naive((n + 15)/16, (n + 15)/16);
    start = clock();
    symm_mult_kernel_naive<<<grid_naive, block_naive>>>(d_A, n, k, d_C);
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_naive, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    double t_naive = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // Optimized kernel
    dim3 block_optim(32, 32);
    dim3 grid_optim((n + 31)/32, (n + 31)/32);
    start = clock();
    symm_mult_kernel_optimized<<<grid_optim, block_optim>>>(d_A, n, k, d_C);
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_optim, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    double t_optim = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    start = clock();
    symm_mult_gpu_cublas(h_A, n, k, h_C_cublas, handle);
    double t_cublas = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // Validation
    float tolerance = 1e-4;
    int valid_naive = compare_matrices(h_C_cpu, h_C_naive, n*n, tolerance);
    int valid_optim = compare_matrices(h_C_cpu, h_C_optim, n*n, tolerance);
    int valid_cublas = compare_matrices(h_C_cpu, h_C_cublas, n*n, tolerance);

    printf("CPU Time:    %7.2f ms\n", cpu_time);
    printf("Naive GPU:   %7.2f ms (Speedup: %5.1fx)\n", t_naive, cpu_time/t_naive);
    printf("Optim GPU:   %7.2f ms (Speedup: %5.1fx)\n", t_optim, cpu_time/t_optim);
    printf("cuBLAS:      %7.2f ms (Speedup: %5.1fx)\n", t_cublas, cpu_time/t_cublas);
    printf("Validation:  Naive=%s  Optim=%s  cuBLAS=%s\n",
           valid_naive ? "PASS" : "FAIL",
           valid_optim ? "PASS" : "FAIL",
           valid_cublas ? "PASS" : "FAIL");

    free(h_A);
    free(h_C_cpu);
    free(h_C_naive);
    free(h_C_optim);
    free(h_C_cublas);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}

int main() {
    run_test(256, 128);
    run_test(512, 256);
    run_test(1024, 512);
    return 0;
}