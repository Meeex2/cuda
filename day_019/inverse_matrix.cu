#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

#define BLOCK_SIZE 16
#define TOL 1e-4

void matrix_inverse_cpu(float* A, int n) {
    std::vector<float> aug(n * 2 * n, 0);
    
    
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j)
            aug[i * 2 * n + j] = A[i * n + j];
        aug[i * 2 * n + n + i] = 1.0f;
    }
    for(int col = 0; col < n; ++col) {
        
        int max_row = col;
        for(int i = col+1; i < n; ++i)
            if(fabs(aug[i * 2 * n + col]) > fabs(aug[max_row * 2 * n + col]))
                max_row = i;
        
        if(max_row != col)
            for(int j = col; j < 2 * n; ++j)
                std::swap(aug[col * 2 * n + j], aug[max_row * 2 * n + j]);
        
        float pivot = aug[col * 2 * n + col];
        for(int j = col; j < 2 * n; ++j)
            aug[col * 2 * n + j] /= pivot;
        
        for(int i = 0; i < n; ++i) {
            if(i != col) {
                float factor = aug[i * 2 * n + col];
                for(int j = col; j < 2 * n; ++j)
                    aug[i * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            A[i * n + j] = aug[i * 2 * n + n + j];
}

__global__ void init_aug_kernel(float* A, float* aug, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < n && col < 2 * n) {
        if(col < n) { 
            aug[row * 2 * n + col] = A[row * n + col];
        } else {      
            aug[row * 2 * n + col] = (col == row + n) ? 1.0f : 0.0f;
        }
    }
}
__global__ void swap_rows_kernel(float* aug, int n, int col, int max_row) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j >= 2 * n || col == max_row) return;
    
    float temp = aug[col * 2 * n + j];
    aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
    aug[max_row * 2 * n + j] = temp;
}
__global__ void normalize_kernel(float* aug, int n, int col) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + col;
    if(j >= 2 * n) return;
    float pivot = aug[col * 2 * n + col];
    if(pivot != 0.0f && j >= col)
        aug[col * 2 * n + j] /= pivot;
}
__global__ void eliminate_kernel(float* aug, int n, int col) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + col;
    
    if(row >= n || j >= 2 * n || row == col) return;
    
    float factor = aug[row * 2 * n + col];
    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
}

void matrix_inverse_gpu(float* d_A, int n) {
    float *d_aug;
    cudaMalloc(&d_aug, n * 2 * n * sizeof(float));
    
    dim3 init_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 init_grid(
        (2 * n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    init_aug_kernel<<<init_grid, init_block>>>(d_A, d_aug, n);
    cudaDeviceSynchronize();
    for(int col = 0; col < n; ++col) {
        
        int num_rows = n - col;
        std::vector<float> h_col(num_rows);
        cudaMemcpy(&h_col[0],
                   d_aug + col * 2 * n + col,
                   num_rows * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        int max_row = col;
        float max_val = fabs(h_col[0]);
        for (int i = 1; i < num_rows; i++) {
            float val = fabs(h_col[i]);
            if(val > max_val) {
                max_val = val;
                max_row = col + i;
            }
        }
        
        dim3 swap_block(256);
        dim3 swap_grid((2 * n + 255) / 256);
        swap_rows_kernel<<<swap_grid, swap_block>>>(d_aug, n, col, max_row);
        cudaDeviceSynchronize();
        
        normalize_kernel<<<(2 * n - col + 255) / 256, 256>>>(d_aug, n, col);
        cudaDeviceSynchronize();
        
        
        dim3 elim_block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 elim_grid(
            (n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
            ((2 * n - col) + BLOCK_SIZE - 1) / BLOCK_SIZE
        );
        eliminate_kernel<<<elim_grid, elim_block>>>(d_aug, n, col);
        cudaDeviceSynchronize();
    }
    
    
    
    cudaMemcpy2D(d_A, n * sizeof(float),
                 d_aug + n, 2 * n * sizeof(float),
                 n * sizeof(float), n,
                 cudaMemcpyDeviceToDevice);
    
    cudaFree(d_aug);
}


void test_matrix_inverse(int n) {
    std::vector<float> h_A(n * n);
    std::vector<float> h_A_inv_cpu(n * n);
    std::vector<float> h_A_inv_gpu(n * n);
    
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);
    
    for(int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for(int j = 0; j < n; ++j) {
            h_A[i * n + j] = dist(gen);
            sum += fabs(h_A[i * n + j]);
        }
        h_A[i * n + i] = sum + 1.0f; 
    }
    
    
    h_A_inv_cpu = h_A;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrix_inverse_cpu(h_A_inv_cpu.data(), n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float>(end_cpu - start_cpu).count();
    
    float *d_A;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrix_inverse_gpu(d_A, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(h_A_inv_gpu.data(), d_A, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time /= 1000.0f; 
    
    
    float max_error = 0.0f;
    for(int i = 0; i < n * n; ++i)
        max_error = fmaxf(max_error, fabsf(h_A_inv_cpu[i] - h_A_inv_gpu[i]));
    
    std::cout << "Matrix Size: " << n << "x" << n << "\n";
    std::cout << "Max Error: " << max_error << "\n";
    std::cout << "CPU Time: " << cpu_time << " s\n";
    std::cout << "GPU Time: " << gpu_time << " s\n";
    std::cout << "Speedup: " << cpu_time / gpu_time << "x\n\n";
    
    cudaFree(d_A);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
int main() {
    test_matrix_inverse(256);
    test_matrix_inverse(512);
    test_matrix_inverse(1024);
    return 0;
}
