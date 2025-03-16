#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdint>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

// Block size for quantization (elements per block)
#define BLOCK_SIZE 64

// CUDA kernel for 4-bit quantization
__global__ void quantize_q4_kernel(const float* input, uint8_t* quantized, float* scales, int num_elements) {
    __shared__ float s_input[BLOCK_SIZE];
    __shared__ float smax[BLOCK_SIZE];
    __shared__ uint8_t q_shared[BLOCK_SIZE];
    __shared__ float s_scale;

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int element_start = block_idx * BLOCK_SIZE;
    int element_end = (element_start + BLOCK_SIZE < num_elements) ? element_start + BLOCK_SIZE : num_elements;

    // Load input into shared memory
    if (element_start + thread_idx < num_elements) {
        s_input[thread_idx] = input[element_start + thread_idx];
    } else {
        s_input[thread_idx] = 0.0f;
    }
    __syncthreads();

    // Compute absolute max
    smax[thread_idx] = fabsf(s_input[thread_idx]);
    __syncthreads();

    // Parallel max reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            smax[thread_idx] = fmaxf(smax[thread_idx], smax[thread_idx + stride]);
        }
        __syncthreads();
    }

    // Compute scale
    if (thread_idx == 0) {
        float absmax = smax[0];
        s_scale = (absmax == 0.0f) ? 1.0f : (absmax / 7.0f);
        scales[block_idx] = s_scale;
    }
    __syncthreads();

    // Quantize elements
    uint8_t q = 8; // Default to zero (8 -> 0 after dequantization)
    if (element_start + thread_idx < num_elements) {
        float val = s_input[thread_idx];
        float q_val = roundf(val / s_scale);
        q_val = fmaxf(fminf(q_val, 7.0f), -8.0f);
        q = static_cast<uint8_t>(q_val + 8.0f);
    }
    q_shared[thread_idx] = q;
    __syncthreads();

    // Pack 4-bit values into 8-bit storage
    if (thread_idx < BLOCK_SIZE / 2) {
        uint8_t high = q_shared[2 * thread_idx];
        uint8_t low = q_shared[2 * thread_idx + 1];
        quantized[block_idx * (BLOCK_SIZE / 2) + thread_idx] = (high << 4) | low;
    }
}

