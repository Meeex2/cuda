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

// CUDA kernel for 4-bit dequantization
__global__ void dequantize_q4_kernel(const uint8_t* quantized, const float* scales, float* output, int num_elements) {
    __shared__ float s_scale;
    __shared__ uint8_t q_packed[BLOCK_SIZE / 2];
    __shared__ uint8_t q_unpacked[BLOCK_SIZE];

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int element_start = block_idx * BLOCK_SIZE;
    int element_end = min(element_start + BLOCK_SIZE, num_elements);

    // Load scale
    if (thread_idx == 0) {
        s_scale = scales[block_idx];
    }
    __syncthreads();

    // Load packed data
    if (thread_idx < BLOCK_SIZE / 2) {
        q_packed[thread_idx] = quantized[block_idx * (BLOCK_SIZE / 2) + thread_idx];
    }
    __syncthreads();

    // Unpack 4-bit values
    if (thread_idx < BLOCK_SIZE / 2) {
        uint8_t packed = q_packed[thread_idx];
        q_unpacked[2 * thread_idx] = (packed >> 4) & 0x0F;
        q_unpacked[2 * thread_idx + 1] = packed & 0x0F;
    }
    __syncthreads();

    // Dequantize
    if (element_start + thread_idx < num_elements) {
        uint8_t q = q_unpacked[thread_idx];
        output[element_start + thread_idx] = (static_cast<float>(q) - 8.0f) * s_scale;
    }
}

// CPU reference implementation for 4-bit quantization
void quantize_q4_cpu(const float* input, uint8_t* quantized, float* scales, int num_elements) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int b = 0; b < num_blocks; ++b) {
        int start = b * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, num_elements);
        float absmax = 0.0f;

        // Compute absmax
        for (int i = start; i < end; ++i) {
            absmax = std::max(absmax, std::abs(input[i]));
        }

        // Compute scale
        float scale = (absmax == 0.0f) ? 1.0f : (absmax / 7.0f);
        scales[b] = scale;

        // Quantize and pack
        std::vector<uint8_t> qs(BLOCK_SIZE, 8);
        for (int i = start; i < end; ++i) {
            float q_val = std::round(input[i] / scale);
            q_val = std::max(std::min(q_val, 7.0f), -8.0f);
            qs[i - start] = static_cast<uint8_t>(q_val + 8.0f);
        }

        // Pack into bytes
        for (int i = 0; i < BLOCK_SIZE / 2; ++i) {
            quantized[b * (BLOCK_SIZE / 2) + i] = (qs[2 * i] << 4) | qs[2 * i + 1];
        }
    }
}



