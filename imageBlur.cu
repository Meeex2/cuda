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

int main() {
    
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    int kernelSize = KERNEL_SIZE;
    
    size_t imageSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);
    float *h_inputImage = (float*)malloc(imageSize);
    float *h_outputImage = (float*)malloc(imageSize);
    float *h_kernel = (float*)malloc(kernelSizeBytes);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_inputImage[y * width + x] = static_cast<float>(x + y) / (width + height); 
        }
    }
    
    float kernel[9] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,
        2.0f / 16, 4.0f / 16, 2.0f / 16,
        1.0f / 16, 2.0f / 16, 1.0f / 16
    };
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        h_kernel[i] = kernel[i];
    }
    
    float *d_inputImage, *d_outputImage, *d_kernel;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);
    
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSizeBytes, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((IMAGE_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IMAGE_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    imageBlurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_inputImage, d_outputImage, d_kernel, width, height, kernelSize);
    
    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);
    
    printf("Blurred image pixel at (128, 128): %f\n", h_outputImage[128 * width + 128]);
    printf("Output image block around (128, 128):\n");
    for (int y = 126; y <= 130; y++) {
        for (int x = 126; x <= 130; x++) {
            printf("%f ", h_outputImage[y * width + x]);
        }
        printf("\n");
    }
    
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_kernel);
    
    free(h_inputImage);
    free(h_outputImage);
    free(h_kernel);
    return 0;
}
