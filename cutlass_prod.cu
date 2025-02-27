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

