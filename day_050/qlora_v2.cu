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


// ------------------------------------------------------------------
// Kernel: NF4 dequantization
__global__ void dequantize_nf4_kernel(const uint8_t* quantized, const float* scales, float* output, int num_elements) {
    __shared__ float s_scale;
    __shared__ uint8_t s_packed[BLOCK_SIZE/2];
    __shared__ uint8_t s_indices[BLOCK_SIZE];

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int element_idx = block_idx * BLOCK_SIZE + thread_idx;
    
    // Load scale for this block
    if (thread_idx == 0) {
        s_scale = scales[block_idx];
    }
    
    // Load packed quantized data
    if (thread_idx < BLOCK_SIZE/2) {
        s_packed[thread_idx] = quantized[block_idx * (BLOCK_SIZE/2) + thread_idx];
    }
    __syncthreads();
    
    // Unpack two 4-bit indices per byte
    if (thread_idx < BLOCK_SIZE/2) {
        uint8_t packed = s_packed[thread_idx];
        s_indices[2 * thread_idx] = (packed >> 4) & 0x0F;
        s_indices[2 * thread_idx + 1] = packed & 0x0F;
    }
    __syncthreads();
    
    // Dequantize: map index to level and scale
    if (element_idx < num_elements) {
        uint8_t idx = s_indices[thread_idx];
        output[element_idx] = d_nf4_levels[idx] * s_scale;
    }
}

// ------------------------------------------------------------------
// Kernel: double quantization of scales
__global__ void double_quantize_scales_kernel(const float* scales, uint8_t* q_scales, float global_max, int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_blocks) {
        float norm = (global_max > 0) ? scales[idx] / global_max : 0.0f;
        uint8_t q = static_cast<uint8_t>(roundf(norm * 15.0f));
        q_scales[idx] = q;
    }
}

// Kernel for dequantizing scales
__global__ void double_dequantize_scales_kernel(const uint8_t* q_scales, float* scales_dq, float global_max, int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_blocks) {
        scales_dq[idx] = (q_scales[idx] / 15.0f) * global_max;
    }
}

// ------------------------------------------------------------------
// CPU reference implementation for NF4 quantization
void quantize_nf4_cpu(const float* input, uint8_t* quantized, float* scales, int num_elements) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int b = 0; b < num_blocks; ++b) {
        int start = b * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, num_elements);
        
        // 1. Compute absmax for the block
        float absmax = 0.0f;
        for (int i = start; i < end; ++i) {
            absmax = std::max(absmax, std::abs(input[i]));
        }
        
        // 2. Compute scale factor (absmax or 1 if absmax is 0)
        float scale = (absmax == 0.0f) ? 1.0f : absmax;
        scales[b] = scale;
        
        // 3. Quantize each element in the block
        std::vector<uint8_t> indices(BLOCK_SIZE, 0);
        for (int i = start; i < end; ++i) {
            float norm = input[i] / scale;
            
            // Find closest NF4 level
            float min_diff = std::abs(norm - h_nf4_levels[0]);
            int best_idx = 0;
            
            for (int j = 1; j < 16; ++j) {
                float diff = std::abs(norm - h_nf4_levels[j]);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_idx = j;
                }
            }
            
            indices[i - start] = static_cast<uint8_t>(best_idx);
        }
        
        // 4. Pack two 4-bit indices per byte
        for (int i = 0; i < BLOCK_SIZE/2; ++i) {
            uint8_t high = indices[2 * i];
            uint8_t low = indices[2 * i + 1];
            quantized[b * (BLOCK_SIZE/2) + i] = (high << 4) | (low & 0x0F);
        }
    }
}

