#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// Function to initialize matrix with random values
void initialize_matrix(float* matrix, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

// CPU reference implementation of matrix multiplication
void cpu_matrix_multiply(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Validation function
bool validate_results(const float* cpu, const float* gpu, int size, float tolerance = 1e-3) {
    for (int i = 0; i < size; ++i) {
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
    // Matrix dimensions
    const int M = 1024;  // Rows of A and C
    const int N = 1024;  // Columns of B and C
    const int K = 1024;  // Columns of A and rows of B

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_gpu(M * N, 0.0f);

    // Initialize matrices with random values
    initialize_matrix(h_A.data(), M, K);
    initialize_matrix(h_B.data(), K, N);

    // Run CPU version
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matrix_multiply(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUTLASS GEMM operation
    using Gemm = cutlass::gemm::device::Gemm<
        float,                                   // Element type for A
        cutlass::layout::RowMajor,              // Layout of A
        float,                                   // Element type for B
        cutlass::layout::RowMajor,              // Layout of B
        float,                                   // Element type for C
        cutlass::layout::RowMajor,              // Layout of C
        float,                                   // Accumulator type
        cutlass::arch::OpClassSimt,             // Operation class
        cutlass::arch::Sm75                     // CUDA architecture (Turing)
    >;

    // Define GEMM arguments
    typename Gemm::Arguments args(
        {M, N, K},                              // Problem dimensions
        {d_A, K},                               // Matrix A (row-major)
        {d_B, N},                               // Matrix B (row-major)
        {d_C, N},                               // Matrix C (row-major)
        {d_C, N},                               // Matrix D (row-major)
        {1.0f, 0.0f}                            // Alpha and beta
    );

    // Run GPU version
    Gemm gemm_op;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cutlass::Status status = gemm_op(args);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  // Convert to seconds

    // Check for errors
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed with error: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    // Copy GPU results back
    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate results
    bool validation = validate_results(h_C_cpu.data(), h_C_gpu.data(), M * N);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;

    // Print timings
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

