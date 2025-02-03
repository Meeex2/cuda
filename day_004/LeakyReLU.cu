#include <stdio.h>
#include <cuda_runtime.h>
__global__ void leakyReluKernel(float* data, int size, float alpha) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        if (data[idx] >= 0) {
            
            data[idx] = data[idx];
        } else {
            
            data[idx] = alpha * data[idx];
        }
    }
}
void leakyRelu(float* data, int size, float alpha) {
    float* d_data;
    
    cudaMalloc((void**)&d_data, size * sizeof(float));
    
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    leakyReluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size, alpha);
    
    cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
}
int main() {
    const int size = 10; 
    float data[size] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -3.0f, 3.0f, -4.0f, 4.0f, -5.0f};
    float alpha = 0.01f; 
    
    printf("Input data:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
    
    leakyRelu(data, size, alpha);
    
    printf("Output data after Leaky ReLU:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
    return 0;
}
