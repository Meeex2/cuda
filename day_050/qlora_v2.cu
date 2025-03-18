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

// ------------------------------------------------------------------
// Main testing function
int main() {
    const int num_elements = 1024;
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("Testing NF4 quantization with %d elements in %d blocks\n", num_elements, num_blocks);
    
    // Host vectors
    std::vector<float> h_input(num_elements);
    std::vector<uint8_t> h_quantized_nf4_cpu(num_elements / 2, 0);
    std::vector<uint8_t> h_quantized_nf4_gpu(num_elements / 2, 0);
    std::vector<float> h_scales_nf4_cpu(num_blocks);
    std::vector<float> h_scales_nf4_gpu(num_blocks);
    std::vector<uint8_t> h_qscales(num_blocks);
    std::vector<float> h_scales_dq(num_blocks);
    std::vector<float> h_output_nf4_gpu(num_elements);
    
    // Seed random number generator for reproducibility
    srand(42);
    
    // Generate random input data (uniform [-1,1])
    for (int i = 0; i < num_elements; ++i) {
        h_input[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_scales, *d_output;
    uint8_t *d_quantized, *d_qscales;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_quantized, (num_elements / 2) * sizeof(uint8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scales, num_blocks * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_qscales, num_blocks * sizeof(uint8_t)));
    
    // Initialize device memory to zero (important for quantization)
    CHECK_CUDA_ERROR(cudaMemset(d_quantized, 0, (num_elements / 2) * sizeof(uint8_t)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run GPU quantization
    printf("Running GPU NF4 quantization...\n");
    quantize_nf4_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_quantized, d_scales, num_elements);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy GPU results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_quantized_nf4_gpu.data(), d_quantized, (num_elements / 2) * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_scales_nf4_gpu.data(), d_scales, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Run CPU reference quantization
    printf("Running CPU reference NF4 quantization...\n");
    quantize_nf4_cpu(h_input.data(), h_quantized_nf4_cpu.data(), h_scales_nf4_cpu.data(), num_elements);
    
    // Compare CPU and GPU results
    bool error = false;
    int scale_mismatches = 0;
    int data_mismatches = 0;
    
    printf("Checking scale factors...\n");
    for (int i = 0; i < num_blocks; ++i) {
        if (fabs(h_scales_nf4_cpu[i] - h_scales_nf4_gpu[i]) > 1e-5) {
            if (scale_mismatches < 10) {
                printf("Scale mismatch at block %d: CPU %f vs GPU %f\n", i, h_scales_nf4_cpu[i], h_scales_nf4_gpu[i]);
            }
            scale_mismatches++;
            error = true;
        }
    }
    
    printf("Checking quantized data...\n");
    for (int i = 0; i < num_elements / 2; ++i) {
        if (h_quantized_nf4_cpu[i] != h_quantized_nf4_gpu[i]) {
            if (data_mismatches < 10) {
                printf("Quantized data mismatch at byte %d: CPU 0x%02x vs GPU 0x%02x\n", i, h_quantized_nf4_cpu[i], h_quantized_nf4_gpu[i]);
            }
            data_mismatches++;
            error = true;
        }
    }
    
    printf("Total mismatches: %d scale values, %d quantized bytes\n", scale_mismatches, data_mismatches);
    
    // Double quantization of scales
    printf("\nTesting double quantization of scales...\n");
    float global_max = 0.0f;
    for (int i = 0; i < num_blocks; ++i) {
        global_max = std::max(global_max, h_scales_nf4_gpu[i]);
    }
    
    int threads_per_block = 256;
    int blocks_scales = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    double_quantize_scales_kernel<<<blocks_scales, threads_per_block>>>(d_scales, d_qscales, global_max, num_blocks);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_qscales.data(), d_qscales, num_blocks * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    double_dequantize_scales_kernel<<<blocks_scales, threads_per_block>>>(d_qscales, d_scales, global_max, num_blocks);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_scales_dq.data(), d_scales, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Double quantization results:\n");
    for (int i = 0; i < num_blocks; ++i) {
        printf("Block %d: Original scale = %f, quantized scale = %u, dequantized scale = %f\n",
               i, h_scales_nf4_gpu[i], h_qscales[i], h_scales_dq[i]);
    }
    
    // Dequantize data using the double-quantized scales
    printf("\nTesting NF4 dequantization...\n");
    dequantize_nf4_kernel<<<num_blocks, BLOCK_SIZE>>>(d_quantized, d_scales, d_output, num_elements);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_nf4_gpu.data(), d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute RMSE to verify quality of quantization+dequantization
    float rmse = 0.0f;
    for (int i = 0; i < num_elements; ++i) {
        float diff = h_input[i] - h_output_nf4_gpu[i];
        rmse += diff * diff;
    }
    rmse = sqrtf(rmse / num_elements);
    printf("NF4 Dequantization RMSE: %e\n", rmse);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_quantized);
    cudaFree(d_scales);
    cudaFree(d_output);
    cudaFree(d_qscales);
    
    return error ? 1 : 0;
}
