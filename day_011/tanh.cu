#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ARRAY_SIZE 1024

__global__ void tanhKernel(const float* input, float* output, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        
        output[idx] = tanh(input[idx]); 
    }
}
int main() {
    int size = ARRAY_SIZE;
    size_t arraySize = size * sizeof(float);
    
    float *h_input = (float*)malloc(arraySize);
    float *h_output = (float*)malloc(arraySize);
    
    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, arraySize);
    cudaMalloc(&d_output, arraySize);
    
    cudaMemcpy(d_input, h_input, arraySize, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    
    cudaMemcpy(h_output, d_output, arraySize, cudaMemcpyDeviceToHost);
    
    printf("Tanh of the first 5 elements by Device:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    printf("Tanh of the first 5 elements by Host:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", tanh(h_input[i]));
    }
    printf("\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    free(h_input);
    free(h_output);
    return 0;
}