#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define VECTOR_SIZE 1024
#define THREADS_PER_BLOCK 256
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

__global__ void reduceSumKernel(const float* input, float* output, int size) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedData[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

__global__ void normalizeKernel(float* output, int size, float sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = output[idx] / sum;
    }
}

float deviceSum(const float* d_array, int size) {
    int threads = THREADS_PER_BLOCK;
    int blocks = (size + threads - 1) / threads;
    
    float* d_partialSums;
    CUDA_CHECK(cudaMalloc(&d_partialSums, blocks * sizeof(float)));
    
    reduceSumKernel<<<blocks, threads, threads * sizeof(float)>>>(d_array, d_partialSums, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float h_sum = 0.0f;
    if (blocks > 1) {
        float* h_partialSums = (float*)malloc(blocks * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < blocks; i++) {
            h_sum += h_partialSums[i];
        }
        free(h_partialSums);
    } else {
        CUDA_CHECK(cudaMemcpy(&h_sum, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_partialSums));
    return h_sum;
}

void testSoftmax(const float* output, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += output[i];
        if (output[i] < 0.0f || output[i] > 1.0f) {
            printf("Error: Softmax probability out of range at index %d: %f\n", i, output[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Sum of softmax probabilities: %f (should be ~1.0)\n", sum);
    if (fabs(sum - 1.0f) > 1e-5) {
        printf("Error: Softmax probabilities do not sum to 1.0\n");
        exit(EXIT_FAILURE);
    }
    printf("Softmax output is valid!\n");
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
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_output, vectorSize));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, vectorSize, cudaMemcpyHostToDevice));
    
    int threads = THREADS_PER_BLOCK;
    int blocks = (size + threads - 1) / threads;
    softmaxKernel<<<blocks, threads>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float sum = deviceSum(d_output, size);
    
    normalizeKernel<<<blocks, threads>>>(d_output, size, sum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, vectorSize, cudaMemcpyDeviceToHost));
    
    testSoftmax(h_output, size);
    
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
