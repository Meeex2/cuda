%%writefile flash_attention.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SEQ_LEN 64
#define DIM 64
#define TILE_SIZE 16

__global__ void flashAttentionKernel(const float* Q, const float* K, const float* V, float* output, int seqLen, int dim) {
    
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float local_sum = 0.0f;
    float local_max = -INFINITY;
    
    float out_val = 0.0f;
    
    for (int t = 0; t < seqLen / TILE_SIZE; t++) {
        
        Q_tile[ty][tx] = (row < seqLen && t * TILE_SIZE + tx < dim) ? Q[row * dim + t * TILE_SIZE + tx] : 0.0f;
        
        K_tile[ty][tx] = (col < seqLen && t * TILE_SIZE + ty < dim) ? K[col * dim + t * TILE_SIZE + ty] : 0.0f;
        
        V_tile[ty][tx] = (t * TILE_SIZE + ty < seqLen && tx < dim) ? V[(t * TILE_SIZE + ty) * dim + tx] : 0.0f;
        __syncthreads();
        
        float score = 0.0f;
        for (int i = 0; i < TILE_SIZE; i++) {
            score += Q_tile[ty][i] * K_tile[i][tx]; 
        }
        score /= sqrtf(dim);
        
        local_max = fmaxf(local_max, score);
        local_sum += expf(score - local_max);
        
        for (int i = 0; i < TILE_SIZE; i++) {
            out_val += expf(score - local_max) * V_tile[i][tx]; 
        }
        __syncthreads();
    }
    
    out_val /= local_sum;
    
    if (row < seqLen && col < dim) {
        output[row * dim + col] = out_val;
    }
}
