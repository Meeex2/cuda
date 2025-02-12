%%writefile flash.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstdio>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define BLOCK_SIZE 128
#define TILE_SIZE_M 64  
#define TILE_SIZE_N 64  
#define HEAD_DIM 64     

__global__ void flash_attention_forward(
    const float* Q,    
    const float* K,
    const float* V,
    float* O,          
    const int seq_len,
    const float scale  
) {
    
    __shared__ float Q_tile[TILE_SIZE_M][HEAD_DIM];
    __shared__ float K_tile[TILE_SIZE_N][HEAD_DIM];
    __shared__ float V_tile[TILE_SIZE_N][HEAD_DIM];
    
    const int tid = threadIdx.x;
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int m_offset = blockIdx.x * TILE_SIZE_M;
    
    
    float O_local[HEAD_DIM] = {0.0f};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    for (int n_start = 0; n_start < seq_len; n_start += TILE_SIZE_N) {
        
        if (tid < TILE_SIZE_M && (m_offset + tid) < seq_len) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                
                Q_tile[tid][d] = Q[((batch * gridDim.y + head) * seq_len + (m_offset + tid)) * HEAD_DIM + d];
            }
        }
        
        
        if (tid < TILE_SIZE_N && (n_start + tid) < seq_len) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                K_tile[tid][d] = K[((batch * gridDim.y + head) * seq_len + (n_start + tid)) * HEAD_DIM + d];
            }
        }
        __syncthreads();
        
        
        
        
        for (int n = 0; n < TILE_SIZE_N; ++n) {
            
            if (n_start + n < seq_len && tid < TILE_SIZE_M && (m_offset + tid) < seq_len) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    s += Q_tile[tid][d] * K_tile[n][d];
                }
                s *= scale;
                
                float old_max = max_val;
                max_val = fmaxf(max_val, s);
                sum_exp = sum_exp * expf(old_max - max_val) + expf(s - max_val);
            }
        }
        __syncthreads();
        
        if (tid < TILE_SIZE_N && (n_start + tid) < seq_len) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                V_tile[tid][d] = V[((batch * gridDim.y + head) * seq_len + (n_start + tid)) * HEAD_DIM + d];
            }
        }
        __syncthreads();
        
        
        
        for (int n = 0; n < TILE_SIZE_N; ++n) {
            if (n_start + n < seq_len && tid < TILE_SIZE_M && (m_offset + tid) < seq_len) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    s += Q_tile[tid][d] * K_tile[n][d];
                }
                s *= scale;
                float p = expf(s - max_val) / sum_exp;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_local[d] += p * V_tile[n][d];
                }
            }
        }
        __syncthreads();
    }
    
    if (tid < TILE_SIZE_M && (m_offset + tid) < seq_len) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O[((batch * gridDim.y + head) * seq_len + (m_offset + tid)) * HEAD_DIM + d] = O_local[d];
        }
    }
}

void compute_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    dim3 grid_dim(
        (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M,  
        num_heads,                                  
        batch_size                                  
    );
    dim3 block_dim(BLOCK_SIZE);
    
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    flash_attention_forward<<<grid_dim, block_dim>>>( Q, K, V, O, seq_len, scale );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void compute_attention_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            
            for (int i = 0; i < seq_len; i++) {
                
                float max_val = -INFINITY;
                std::vector<float> scores(seq_len, 0.0f);
                for (int j = 0; j < seq_len; j++) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        int idx_q = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                        int idx_k = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                        s += Q[idx_q] * K[idx_k];
                    }
                    s *= scale;
                    scores[j] = s;
                    if (s > max_val) max_val = s;
                }
                
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = expf(scores[j] - max_val);
                    sum_exp += scores[j];
                }
                
                for (int d = 0; d < head_dim; d++) {
                    float out = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        int idx_v = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                        out += (scores[j] / sum_exp) * V[idx_v];
                    }
                    int idx_o = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                    O[idx_o] = out;
                }
            }
        }
    }
}