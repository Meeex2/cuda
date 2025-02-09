#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE (1 << 20) 


__global__ void geluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); 
        output[idx] = x * cdf;
    }
}

void geluCPU(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); 
        output[i] = x * cdf;
    }
}
int main() {
    int size = ARRAY_SIZE;
    size_t arraySize = size * sizeof(float);
    
    float *h_input = (float*)malloc(arraySize);
    float *h_output_cpu = (float*)malloc(arraySize);
    float *h_output_gpu = (float*)malloc(arraySize);
    
    for (int i = 0; i < size; i++) {
        h_input[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX - 0.5f); 
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, arraySize);
    cudaMalloc(&d_output, arraySize);
    
    cudaMemcpy(d_input, h_input, arraySize, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    cudaMemcpy(h_output_gpu, d_output, arraySize, cudaMemcpyDeviceToHost);
    
    clock_t cpuStart = clock();
    geluCPU(h_input, h_output_cpu, size);
    clock_t cpuEnd = clock();
    float cpuTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC; 

    
    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Results match! GELU implementation is correct.\n");
    } else {
        printf("Results do not match! Check the implementation.\n");
    }
    
    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("GPU Time: %.3f ms\n", gpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    return 0;
}