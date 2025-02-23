#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

// Tile size for shared memory
const int TILE_SIZE = 128;

// CUDA quantization kernel with shared memory and tiling
__global__ void quantize_kernel(const float* input, uint8_t* output, 
                               float scale, uint8_t zero_point, 
                               int num_elements) {
    __shared__ float shared_input[TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (global_idx < num_elements) {
        shared_input[tid] = input[global_idx];
    }
    __syncthreads();
    
    // Process data in shared memory
    if (global_idx < num_elements) {
        float val = shared_input[tid] / scale + zero_point;
        output[global_idx] = static_cast<uint8_t>(fminf(fmaxf(roundf(val), 0.0f), 255.0f));
    }
}

// CPU reference implementation
void cpu_quantize(const float* input, uint8_t* output,
                 float scale, uint8_t zero_point,
                 int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        float val = input[i] / scale + zero_point;
        output[i] = static_cast<uint8_t>(std::min(std::max(std::round(val), 0.0f), 255.0f));
    }
}

// Validation function
bool validate_quantized(const uint8_t* cpu, const uint8_t* gpu, int size) {
    for (int i = 0; i < size; ++i) {
        if (cpu[i] != gpu[i]) {
            std::cout << "Mismatch at index " << i 
                      << ": CPU=" << static_cast<int>(cpu[i])
                      << ", GPU=" << static_cast<int>(gpu[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int num_elements = 1 << 24;  // 16.7M elements
    const int block_size = TILE_SIZE;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    // Quantization parameters
    const float scale = 0.1f;
    const uint8_t zero_point = 128;

    // Generate input data with fixed seed
    std::vector<float> host_input(num_elements);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> int_dist(-127, 127);
    
    for (int i = 0; i < num_elements; ++i) {
        host_input[i] = int_dist(gen) * scale;  // Exact multiples of scale
    }

    // Allocate device memory
    float *d_input;
    uint8_t *d_output;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_output, num_elements * sizeof(uint8_t));
    
    cudaMemcpy(d_input, host_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Run CPU version
    std::vector<uint8_t> cpu_output(num_elements);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_quantize(host_input.data(), cpu_output.data(), scale, zero_point, num_elements);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    // Run GPU version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    quantize_kernel<<<grid_size, block_size>>>(d_input, d_output, scale, zero_point, num_elements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  // Convert to seconds

    // Copy GPU results back
    std::vector<uint8_t> gpu_output(num_elements);
    cudaMemcpy(gpu_output.data(), d_output, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Validate results
    bool validation = validate_quantized(cpu_output.data(), gpu_output.data(), num_elements);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;

    // Print timings
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}