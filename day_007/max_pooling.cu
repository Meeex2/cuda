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
