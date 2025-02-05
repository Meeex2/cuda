#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SEQ_LEN 64
#define DIM 64

__global__ void attentionScoresKernel(const float* Q, const float* K, float* scores, int seqLen, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seqLen && col < seqLen) {
        float score = 0.0f;
        for (int i = 0; i < dim; i++) {
            score += Q[row * dim + i] * K[col * dim + i]; 
        }
        scores[row * seqLen + col] = score / sqrtf(dim); 
    }
}

__global__ void softmaxKernel(float* scores, int seqLen) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seqLen) {
        float maxVal = -INFINITY;
        for (int i = 0; i < seqLen; i++) {
            maxVal = fmaxf(maxVal, scores[row * seqLen + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < seqLen; i++) {
            scores[row * seqLen + i] = expf(scores[row * seqLen + i] - maxVal);
            sum += scores[row * seqLen + i];
        }
        for (int i = 0; i < seqLen; i++) {
            scores[row * seqLen + i] /= sum;
        }
    }
}

__global__ void weightedSumKernel(const float* scores, const float* V, float* output, int seqLen, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seqLen && col < dim) {
        float sum = 0.0f;
        for (int i = 0; i < seqLen; i++) {
            sum += scores[row * seqLen + i] * V[i * dim + col]; 
        }
        output[row * dim + col] = sum;
    }
}
int main() {
    int seqLen = SEQ_LEN;
    int dim = DIM;
    size_t matrixSize = seqLen * dim * sizeof(float);
    size_t scoresSize = seqLen * seqLen * sizeof(float);
    
    float *h_Q = (float*)malloc(matrixSize);
    float *h_K = (float*)malloc(matrixSize);
    float *h_V = (float*)malloc(matrixSize);
    float *h_scores = (float*)malloc(scoresSize);
    float *h_output = (float*)malloc(matrixSize);
    
    for (int i = 0; i < seqLen * dim; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    float *d_Q, *d_K, *d_V, *d_scores, *d_output;
    cudaMalloc(&d_Q, matrixSize);
    cudaMalloc(&d_K, matrixSize);
    cudaMalloc(&d_V, matrixSize);
    cudaMalloc(&d_scores, scoresSize);
    cudaMalloc(&d_output, matrixSize);
    
    cudaMemcpy(d_Q, h_Q, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, matrixSize, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGridScores((seqLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (seqLen + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 blocksPerGridOutput((dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (seqLen + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    attentionScoresKernel<<<blocksPerGridScores, threadsPerBlock>>>(d_Q, d_K, d_scores, seqLen, dim);
    
    softmaxKernel<<<seqLen, 1>>>(d_scores, seqLen);
    
    weightedSumKernel<<<blocksPerGridOutput, threadsPerBlock>>>(d_scores, d_V, d_output, seqLen, dim);
    
    cudaMemcpy(h_output, d_output, matrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scores, d_scores, scoresSize, cudaMemcpyDeviceToHost);
    
    float scores[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) {
        scores[i] = h_scores[0 * SEQ_LEN + i];
    }
    float V_col[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) {
        V_col[i] = h_V[i * DIM + 0];
    }
    float expectedOutput = 0.0f;
    for (int i = 0; i < SEQ_LEN; i++) {
        expectedOutput += scores[i] * V_col[i];
    }
    printf("Expected output at (0, 0): %f\n", expectedOutput);
    printf("GPU output at (0, 0): %f\n", h_output[0]);
    if (fabs(h_output[0] - expectedOutput) < 1e-5) {
        printf("Verification successful!\n");
    } else {
        printf("Verification failed!\n");
    }
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_output);
    
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_scores);
    free(h_output);
    return 0;
}
