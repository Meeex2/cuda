#include <stdio.h>
#include <cuda_runtime.h>
__global__ void matrixVectorMulKernel(const float* A, const float* x, float* y, int numRows, int numCols) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        
        for (int col = 0; col < numCols; col++) {
            sum += A[row * numCols + col] * x[col];
        }
        
        y[row] = sum;
    }
}
void matrixVectorMul(const float* A, const float* x, float* y, int numRows, int numCols) {
    float *d_A, *d_x, *d_y;
    
    cudaMalloc((void**)&d_A, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&d_x, numCols * sizeof(float));
    cudaMalloc((void**)&d_y, numRows * sizeof(float));
    
    cudaMemcpy(d_A, A, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, numCols * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    
    matrixVectorMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_y, numRows, numCols);
    
    cudaMemcpy(y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}
int main() {
    const int numRows = 4;
    const int numCols = 3;
    
    float A[numRows][numCols] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f},
        {10.0f, 11.0f, 12.0f}
    };
    
    float x[numCols] = {1.0f, 2.0f, 3.0f};
    
    float y[numRows];
    
    matrixVectorMul((float*)A, x, y, numRows, numCols);
    
    printf("Result vector y:\n");
    for (int i = 0; i < numRows; i++) {
        printf("y[%d] = %f\n", i, y[i]);
    }
    return 0;
}
