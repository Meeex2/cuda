#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_SIZE 1024
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void softmaxKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = exp(input[idx]);
    }
}

__global__ void normalizeKernel(float* output, int size, float sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = output[idx] / sum;
    }
}

int main() {
    int size = VECTOR_SIZE;
    size_t vectorSize = size * sizeof(float);

    float *h_input = (float*)malloc(vectorSize);
    float *h_output = (float*)malloc(vectorSize);

    srand(1234); 
    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; 
    }

    
    printf("Input vector (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_input[i]);
    }
    printf("\n");

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_output, vectorSize));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, vectorSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(h_output, d_output, vectorSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
        h_sum += h_output[i];
    }

    printf("Sum of exponentials: %f\n", h_sum);

    normalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, size, h_sum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, vectorSize, cudaMemcpyDeviceToHost));

    printf("Softmax probabilities for the first 5 elements:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}

