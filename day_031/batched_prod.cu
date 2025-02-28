#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_runtime.h>

using namespace cutlass;

// Define matrix multiplication parameters
using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

// Define epilogue (alpha * AB + beta * C)
using EpilogueOp = epilogue::thread::LinearCombination<
    ElementC,           // Output type
    1,                  // Elements per access (SIMT)
    ElementAccumulator, // Accumulator type
    ElementAccumulator  // Compute type
>;

// Define the GEMM configuration using SIMT
using Gemm = gemm::device::GemmBatched<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    arch::OpClassSimt,           // Use SIMT
    arch::Sm75,                  // Target architecture (Ampere)
    gemm::GemmShape<128, 128, 8>, // Threadblock shape
    gemm::GemmShape<64, 64, 8>,   // Warp shape
    gemm::GemmShape<1, 1, 1>,     // Instruction shape (SIMT)
    EpilogueOp,
    gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,                            // Pipeline stages
    1,                            // Alignment A 
    1                             // Alignment B 
>;

// CPU reference implementation
void cpu_batched_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int batch_size, int M, int N, int K) {
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int a_idx = b * M * K + i * K + k;
                    int b_idx = b * K * N + k * N + j;
                    sum += A[a_idx] * B[b_idx];
                }
                C[b * M * N + i * N + j] = sum;
            }
        }
    }
}

// Initialize matrices with random values
void initialize_matrices(
    std::vector<float>& A,
    std::vector<float>& B,
    int batch_size, int M, int N, int K) {
    for (int i = 0; i < batch_size * M * K; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < batch_size * K * N; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // Matrix dimensions
    const int batch_size = 4;
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // Allocate host memory
    std::vector<float> h_A(batch_size * M * K);
    std::vector<float> h_B(batch_size * K * N);
    std::vector<float> h_C_cpu(batch_size * M * N);
    std::vector<float> h_C_gpu(batch_size * M * N);

    // Initialize input matrices
    initialize_matrices(h_A, h_B, batch_size, M, N, K);

    // CPU computation and timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_batched_gemm(h_A, h_B, h_C_cpu, batch_size, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

    // GPU computation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, batch_size * M * K * sizeof(float));
    cudaMalloc(&d_B, batch_size * K * N * sizeof(float));
    cudaMalloc(&d_C, batch_size * M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), batch_size * M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), batch_size * K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create GEMM arguments
    typename EpilogueOp::Params epilogue_params(
        1.0f,  // alpha
        0.0f   // beta
    );

    Gemm::Arguments args(
        {M, N, K},                     // Problem size
        {d_A, K},                      // A tensor ref with stride K
        M * K,                         // Batch stride A
        {d_B, N},                      // B tensor ref with stride N
        K * N,                         // Batch stride B
        {d_C, N},                      // C tensor ref with stride N
        M * N,                         // Batch stride C
        {d_C, N},                      // D tensor ref with stride N
        M * N,                         // Batch stride D
        epilogue_params,               // Epilogue parameters
        batch_size                     // Batch count
    );

    // Initialize GEMM object
    Gemm gemm_op;

    // GPU computation and timing
    auto gpu_start = std::chrono::high_resolution_clock::now();
    auto status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {  
        std::cerr << "Failed to initialize GEMM: " << int(status) << std::endl;
        return 1;
    }
    
    status = gemm_op.run();
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);

    if (status != cutlass::Status::kSuccess) {  
        std::cerr << "GEMM failed: " << int(status) << std::endl;
        return 1;
    }

    // Copy results back to host
    cudaMemcpy(h_C_gpu.data(), d_C, batch_size * M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    float max_error = 0.0f;
    for (int i = 0; i < batch_size * M * N; ++i) {
        float error = fabs(h_C_cpu[i] - h_C_gpu[i]);
        max_error = std::max(max_error, error);
    }

    // Print results
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}