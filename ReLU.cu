#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for ReLU
__global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

int main() {
    int n = 10; // Number of elements in the array
    size_t size = n * sizeof(float);

    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i - 5); // Example input: [-5, -4, ..., 4]
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    relu_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Input: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output (ReLU): ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}
