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

// CPU reference implementation
void transpose_cpu(float* in, float* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

// Naive GPU kernel (coalesced reads, strided writes)
__global__ void transpose_naive(float* in, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

// Optimized GPU kernel with shared memory
__global__ void transpose_optimized(float* in, float* out, int rows, int cols) {
    __shared__ float tile[32][32+1]; // +1 for padding to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Read into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();
    
    // Write transposed with switched block/thread indices
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < rows && y < cols) {
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Wrapper function for GPU transpose
void transpose_gpu(float* h_in, float* h_out, int rows, int cols, bool optimized) {
    float *d_in, *d_out;
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_in, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, rows * cols * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    if (optimized) {
        transpose_optimized<<<grid, block>>>(d_in, d_out, rows, cols);
    } else {
        transpose_naive<<<grid, block>>>(d_in, d_out, rows, cols);
    }
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

// Test function
void test_transpose(int rows, int cols) {
    // Allocate and initialize host memory
    float* h_in = (float*)malloc(rows * cols * sizeof(float));
    float* h_out_cpu = (float*)malloc(rows * cols * sizeof(float));
    float* h_out_gpu = (float*)malloc(rows * cols * sizeof(float));
    
    for (int i = 0; i < rows * cols; i++) {
        h_in[i] = (float)rand() / RAND_MAX;
    }
    
    // Compute on CPU
    transpose_cpu(h_in, h_out_cpu, rows, cols);
    
    // Compute on GPU (both naive and optimized)
    transpose_gpu(h_in, h_out_gpu, rows, cols, false); // Naive
    transpose_gpu(h_in, h_out_gpu, rows, cols, true);  // Optimized
    
    // Verify results
    float max_diff = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabs(h_out_cpu[i] - h_out_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("Test %dx%d: Max difference = %.6f\n", rows, cols, max_diff);
    assert(max_diff < 1e-5);
    
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
}

// Benchmark function
void benchmark_transpose(int rows, int cols) {
    // Allocate memory
    float* h_in = (float*)malloc(rows * cols * sizeof(float));
    float* h_out = (float*)malloc(rows * cols * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < rows * cols; i++) {
        h_in[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU benchmark
    clock_t start = clock();
    transpose_cpu(h_in, h_out, rows, cols);
    double cpu_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
    
    // GPU benchmarks (with warmup)
    transpose_gpu(h_in, h_out, rows, cols, false); // Warmup
    
    start = clock();
    transpose_gpu(h_in, h_out, rows, cols, false); // Naive
    cudaDeviceSynchronize();
    double gpu_naive_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
    
    start = clock();
    transpose_gpu(h_in, h_out, rows, cols, true); // Optimized
    cudaDeviceSynchronize();
    double gpu_opt_time = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
    
    printf("Size %dx%d: CPU %.2fms, GPU (naive) %.2fms, GPU (opt) %.2fms\n",
           rows, cols, cpu_time, gpu_naive_time, gpu_opt_time);
    printf("Speedup: Naive %.1fx, Optimized %.1fx\n",
           cpu_time/gpu_naive_time, cpu_time/gpu_opt_time);
    
    free(h_in);
    free(h_out);
}

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    // Test cases
    test_transpose(128, 128);
    test_transpose(1024, 1024);
    
    // Benchmark cases
    benchmark_transpose(4096, 4096);
    benchmark_transpose(8192, 8192);
    
    return 0;
}