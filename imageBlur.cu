%%writefile imageBlur.cu
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define KERNEL_SIZE 3

__global__ void imageBlurKernel(const float* inputImage, float* outputImage, const float* kernel, int width, int height, int kernelSize) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float value = 0.0f;
        
        for (int i = -kernelSize / 2; i <= kernelSize / 2; i++) {
            for (int j = -kernelSize / 2; j <= kernelSize / 2; j++) {
                int imageX = x + i;
                int imageY = y + j;
                
                if (imageX < 0) imageX = 0;
                if (imageX >= width) imageX = width - 1;
                if (imageY < 0) imageY = 0;
                if (imageY >= height) imageY = height - 1;
                
                float kernelValue = kernel[(i + kernelSize / 2) * kernelSize + (j + kernelSize / 2)];
                
                value += inputImage[imageY * width + imageX] * kernelValue; 
            }
        }
        
        outputImage[y * width + x] = value; 
    }
}
