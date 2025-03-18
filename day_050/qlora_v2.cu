#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

// Block size for quantization (elements per block)
#define BLOCK_SIZE 64

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ------------------------------------------------------------------
// Define NF4 quantization levels for both host and device
__constant__ float d_nf4_levels[16] = {
    -2.936f, -2.350f, -1.807f, -1.317f,
    -0.878f, -0.484f, -0.126f,  0.234f,
     0.578f,  0.889f,  1.210f,  1.514f,
     1.800f,  2.088f,  2.400f,  2.750f
};

// Host copy of the same levels for CPU calculations
const float h_nf4_levels[16] = {
    -2.936f, -2.350f, -1.807f, -1.317f,
    -0.878f, -0.484f, -0.126f,  0.234f,
     0.578f,  0.889f,  1.210f,  1.514f,
     1.800f,  2.088f,  2.400f,  2.750f
};

// ------------------------------------------------------------------
// Kernel: NF4 quantization (4-bit NormalFloat)
__global__ void quantize_nf4_kernel(const float* input, uint8_t* quantized, float* scales, int num_elements) {
    __shared__ float s_input[BLOCK_SIZE];
    __shared__ float s_absmax[BLOCK_SIZE];
    __shared__ uint8_t s_quantized[BLOCK_SIZE];
    __shared__ float s_scale;

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int element_idx = block_idx * BLOCK_SIZE + thread_idx;
    
    // Initialize shared memory
    s_quantized[thread_idx] = 0;
    
    // 1. Load data into shared memory
    if (element_idx < num_elements) {
        s_input[thread_idx] = input[element_idx];
    } else {
        s_input[thread_idx] = 0.0f;
    }
    __syncthreads();
    
    // 2. Compute absolute maximum for scaling
    s_absmax[thread_idx] = fabsf(s_input[thread_idx]);
    __syncthreads();
    
    // Parallel reduction to find block maximum
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            s_absmax[thread_idx] = fmaxf(s_absmax[thread_idx], s_absmax[thread_idx + stride]);
        }
        __syncthreads();
    }
    
    // 3. Compute scale factor for the block
    if (thread_idx == 0) {
        float absmax = s_absmax[0];
        s_scale = (absmax == 0.0f) ? 1.0f : absmax;
        scales[block_idx] = s_scale;
    }
    __syncthreads();
    
    // 4. Quantize: find closest NF4 level
    if (element_idx < num_elements) {
        float norm = s_input[thread_idx] / s_scale;
        
        // Find closest match in NF4 levels
        float min_diff = fabsf(norm - d_nf4_levels[0]);
        uint8_t best_idx = 0;
        
        for (int i = 1; i < 16; ++i) {
            float diff = fabsf(norm - d_nf4_levels[i]);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
            }
        }
        
        s_quantized[thread_idx] = best_idx;
    }
    __syncthreads();
    
    // 5. Pack two 4-bit indices into one byte
    if (thread_idx < BLOCK_SIZE/2) {
        uint8_t high = s_quantized[2 * thread_idx];
        uint8_t low = s_quantized[2 * thread_idx + 1];
        uint8_t packed = (high << 4) | (low & 0x0F);
        quantized[block_idx * (BLOCK_SIZE/2) + thread_idx] = packed;
    }
}

