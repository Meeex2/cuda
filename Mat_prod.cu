%%writefile Mat_prod.cu
#include <stdio.h>
#include <cmath> 

__global__ void matrixMultiplicationKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float value = 0.0f;
        
        for (int i = 0; i < K; i++) {
            value += A[row * K + i] * B[i * N + col]; 
        }
        
        C[row * N + col] = value; 
    }
}
