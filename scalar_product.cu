#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#include <stdio.h>


__global__ void scalarProductKernel(const float* a, const float* b, float* c, int n) {
    extern __shared__ float sharedMemory[];

    int tid = threadIdx.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedMemory[tid] = (i < n) ? a[i] * b[i] : 0.0f;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMemory[tid] += sharedMemory[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        c[blockIdx.x] = sharedMemory[0];
    }
}

int main() {
    size_t size = n * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc((n / 256) * sizeof(float));

    for (int i = 0; i < n; i++) {
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, (n / 256) * sizeof(float));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    scalarProductKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, (n / 256) * sizeof(float), cudaMemcpyDeviceToHost);

    float finalResult = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        finalResult += h_c[i];
    }

    printf("Scalar product: %f\n", finalResult);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
