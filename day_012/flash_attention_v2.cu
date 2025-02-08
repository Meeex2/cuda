#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define SEQ_LEN 64
#define DIM 64

__global__ void flashAttentionKernel(const float* Q, const float* K, const float* V,
                                       float* output, int seqLen, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int k = blockIdx.y * blockDim.y + threadIdx.y; 
    if(i < seqLen && k < dim) {
        float m = -INFINITY; 
        float l = 0.0f;      
        float acc = 0.0f;    
        
        
        for (int j = 0; j < seqLen; j++) {
            float score = 0.0f;
            
            for (int d = 0; d < dim; d++) {
                score += Q[i * dim + d] * K[j * dim + d];
            }
            score /= sqrtf((float) dim);
            
            
            float m_new = fmaxf(m, score);
            
            acc = acc * expf(m - m_new);
            l = l * expf(m - m_new);
            m = m_new;
            
            float exp_val = expf(score - m);
            l += exp_val;
            acc += exp_val * V[j * dim + k];
        }
        output[i * dim + k] = acc / l;
    }
}
void flashAttentionCPU(const float* Q, const float* K, const float* V,
                       float* output, int seqLen, int dim) {
    for (int i = 0; i < seqLen; i++) {
        
        for (int k = 0; k < dim; k++) {
            float m = -INFINITY;
            float l = 0.0f;
            float acc = 0.0f;
            for (int j = 0; j < seqLen; j++) {
                float score = 0.0f;
                for (int d = 0; d < dim; d++) {
                    score += Q[i * dim + d] * K[j * dim + d];
                }
                score /= sqrtf((float)dim);
                float m_new = (m > score) ? m : score;
                acc = acc * expf(m - m_new);
                l = l * expf(m - m_new);
                m = m_new;
                float exp_val = expf(score - m);
                l += exp_val;
                acc += exp_val * V[j * dim + k];
            }
            output[i * dim + k] = acc / l;
        }
    }
}
int main() {
    int seqLen = SEQ_LEN;
    int dim = DIM;
    size_t matrixSize = seqLen * dim * sizeof(float);
    
    float *h_Q = (float*)malloc(matrixSize);
    float *h_K = (float*)malloc(matrixSize);
    float *h_V = (float*)malloc(matrixSize);
    float *h_output_gpu = (float*)malloc(matrixSize);
    float *h_output_cpu = (float*)malloc(matrixSize);
    
    
    for (int i = 0; i < seqLen * dim; i++) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }
    
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, matrixSize);
    cudaMalloc(&d_K, matrixSize);
    cudaMalloc(&d_V, matrixSize);
    cudaMalloc(&d_output, matrixSize);
    
    cudaMemcpy(d_Q, h_Q, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, matrixSize, cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((seqLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    cudaEventRecord(start);
    flashAttentionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Q, d_K, d_V, d_output, seqLen, dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsedGPU;
    cudaEventElapsedTime(&elapsedGPU, start, stop);
    
    cudaMemcpy(h_output_gpu, d_output, matrixSize, cudaMemcpyDeviceToHost);
    
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    flashAttentionCPU(h_Q, h_K, h_V, h_output_cpu, seqLen, dim);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedCPU = cpu_end - cpu_start;
    
    
    float maxError = 0.0f;
    for (int i = 0; i < seqLen * dim; i++) {
        float error = fabs(h_output_gpu[i] - h_output_cpu[i]);
        if (error > maxError)
            maxError = error;
    }
    
    printf("Flash-Attention: Output matrix element at (0, 0): %f\n", h_output_gpu[0]);
    printf("GPU Time: %f ms\n", elapsedGPU);
    printf("CPU Time: %f ms\n", elapsedCPU.count());
    printf("Max absolute error between GPU and CPU: %f\n", maxError);
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output_gpu);
    free(h_output_cpu);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}