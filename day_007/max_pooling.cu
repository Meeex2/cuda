#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_WIDTH 256
#define INPUT_HEIGHT 256
#define POOL_WINDOW_SIZE 2

__global__ void maxPoolingKernel(const float* input, float* output, int inputWidth, int inputHeight, int poolWindowSize) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int outputWidth = inputWidth / poolWindowSize;
    int outputHeight = inputHeight / poolWindowSize;
    
    if (x < outputWidth && y < outputHeight) {
        float maxVal = -INFINITY; 
        
        for (int i = 0; i < poolWindowSize; i++) {
            for (int j = 0; j < poolWindowSize; j++) {
                
                int inputX = x * poolWindowSize + i;
                int inputY = y * poolWindowSize + j;
                
                if (inputX < inputWidth && inputY < inputHeight) {
                    
                    maxVal = fmaxf(maxVal, input[inputY * inputWidth + inputX]); 
                }
            }
        }
        
        output[y * outputWidth + x] = maxVal; 
    }
}

int main() {
    
    int inputWidth = INPUT_WIDTH;
    int inputHeight = INPUT_HEIGHT;
    int poolWindowSize = POOL_WINDOW_SIZE;
    
    int outputWidth = inputWidth / poolWindowSize;
    int outputHeight = inputHeight / poolWindowSize;
    
    size_t inputSize = inputWidth * inputHeight * sizeof(float);
    size_t outputSize = outputWidth * outputHeight * sizeof(float);
    float *h_input = (float*)malloc(inputSize);
    float *h_output = (float*)malloc(outputSize);
    
    for (int y = 0; y < inputHeight; y++) {
        for (int x = 0; x < inputWidth; x++) {
            h_input[y * inputWidth + x] = static_cast<float>(rand()) / RAND_MAX; 
        }
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
    
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    maxPoolingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, inputWidth, inputHeight, poolWindowSize);
    
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    int x = 64;
    int y = 64;
    int inputX = x * poolWindowSize;
    int inputY = y * poolWindowSize;
    
    float val1 = h_input[inputY * inputWidth + inputX];
    float val2 = h_input[inputY * inputWidth + (inputX + 1)];
    float val3 = h_input[(inputY + 1) * inputWidth + inputX];
    float val4 = h_input[(inputY + 1) * inputWidth + (inputX + 1)];
    
    float expectedMax = fmaxf(fmaxf(val1, val2), fmaxf(val3, val4));
    
    printf("Expected max pooled value at (%d, %d): %f\n", x, y, expectedMax);
    printf("GPU max pooled value at (%d, %d): %f\n", x, y, h_output[y * outputWidth + x]);
    if (abs(h_output[y * outputWidth + x] - expectedMax) < 1e-5) {
        printf("Verification successful!\n");
    } else {
        printf("Verification failed!\n");
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    free(h_input);
    free(h_output);
    return 0;
}
