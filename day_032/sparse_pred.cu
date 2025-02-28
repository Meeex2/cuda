#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for SpMV (CSR format)
__global__ void spmv_csr_kernel(const float* values, const int* col_indices, const int* row_ptr, const float* x, float* y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_indices[i]];
        }
        y[row] = sum;
    }
}

// CPU reference implementation of SpMV (CSR format)
void spmv_csr_cpu(const std::vector<float>& values, const std::vector<int>& col_indices, const std::vector<int>& row_ptr, const std::vector<float>& x, std::vector<float>& y) {
    int num_rows = row_ptr.size() - 1;
    for (int row = 0; row < num_rows; row++) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_indices[i]];
        }
        y[row] = sum;
    }
}

