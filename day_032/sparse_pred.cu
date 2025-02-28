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

// Validation function
bool validate_results(const std::vector<float>& cpu, const std::vector<float>& gpu, float tolerance = 1e-3) {
    for (size_t i = 0; i < cpu.size(); i++) {
        if (std::fabs(cpu[i] - gpu[i]) > tolerance) {
            std::cout << "Mismatch at index " << i 
                      << ": CPU=" << cpu[i] 
                      << ", GPU=" << gpu[i] 
                      << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Sparse matrix dimensions
    const int num_rows = 8192;
    const int num_cols = 8192;
    const int nnz = 8192 * 100;  // Number of non-zero elements

    // Generate random sparse matrix (CSR format)
    std::vector<float> values(nnz);
    std::vector<int> col_indices(nnz);
    std::vector<int> row_ptr(num_rows + 1, 0);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_value(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_col(0, num_cols - 1);

    int nnz_per_row = nnz / num_rows;
    for (int row = 0; row < num_rows; row++) {
        row_ptr[row + 1] = row_ptr[row] + nnz_per_row;
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
            values[i] = dist_value(gen);
            col_indices[i] = dist_col(gen);
        }
    }

    // Generate random dense vector
    std::vector<float> x(num_cols);
    for (int i = 0; i < num_cols; i++) {
        x[i] = dist_value(gen);
    }

    // Allocate host memory for results
    std::vector<float> y_cpu(num_rows, 0.0f);
    std::vector<float> y_gpu(num_rows, 0.0f);

    // Run CPU version
    auto cpu_start = std::chrono::high_resolution_clock::now();
    spmv_csr_cpu(values, col_indices, row_ptr, x, y_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    // Allocate device memory
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptr;

    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_col_indices, nnz * sizeof(int));
    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_x, num_cols * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;

    // Run GPU version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    spmv_csr_kernel<<<grid_size, block_size>>>(d_values, d_col_indices, d_row_ptr, d_x, d_y, num_rows);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  // Convert to seconds

    // Copy GPU results back
    cudaMemcpy(y_gpu.data(), d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate results
    bool validation = validate_results(y_cpu, y_gpu);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;

    // Print timings
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    // Cleanup
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
