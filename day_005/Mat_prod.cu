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

int main() {
    int M = 1024; 
    int K = 1024; 
    int N = 1024; 
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(i) / (M * K); 
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(i) / (K * N); 
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    bool success = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += h_A[i * K + k] * h_B[k * N + j];
            }
            if (abs(h_C[i * N + j] - expected) > 1e-3) { 
                success = false;
                break;
            }
        }
    }
    if (success) {
        printf("Matrix multiplication successful!\n");
    } else {
        printf("Matrix multiplication failed!\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
