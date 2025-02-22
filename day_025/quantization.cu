#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

__global__ void quantize_kernel(const float* input, uint8_t* output, 
                               float scale, uint8_t zero_point, 
                               int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        float val = input[i] / scale + zero_point;
        output[i] = static_cast<uint8_t>(fminf(fmaxf(roundf(val), 0.0f), 255.0f));
    }
}

__global__ void dequantize_kernel(const uint8_t* input, float* output,
                                 float scale, uint8_t zero_point,
                                 int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        output[i] = (input[i] - zero_point) * scale;
    }
}

void cpu_quantize(const float* input, uint8_t* output,
                 float scale, uint8_t zero_point,
                 int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        float val = input[i] / scale + zero_point;
        output[i] = static_cast<uint8_t>(std::min(std::max(std::round(val), 0.0f), 255.0f));
    }
}
void cpu_dequantize(const uint8_t* input, float* output,
                   float scale, uint8_t zero_point,
                   int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        output[i] = (input[i] - zero_point) * scale;
    }
}

bool validate_quantized(const uint8_t* cpu, const uint8_t* gpu, int size) {
    for (int i = 0; i < size; ++i) {
        if (cpu[i] != gpu[i]) {
            std::cout << "Quantization mismatch at " << i 
                      << ": CPU=" << static_cast<int>(cpu[i])
                      << " GPU=" << static_cast<int>(gpu[i]) << std::endl;
            return false;
        }
    }
    return true;
}
bool validate_dequantized(const float* original, const float* dequantized,
                         int size, float scale) {
    const float tolerance = scale / 2.0f + 1e-6f;  
    for (int i = 0; i < size; ++i) {
        if (fabs(original[i] - dequantized[i]) > tolerance) {
            std::cout << "Large error at " << i 
                      << ": Original=" << original[i]
                      << " Dequantized=" << dequantized[i] 
                      << " (Diff: " << fabs(original[i] - dequantized[i]) 
                      << ")" << std::endl;
            return false;
        }
    }
    return true;
}
int main() {
    const int num_elements = 1 << 24;  
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    
    const float scale = 0.1f;
    const uint8_t zero_point = 128;
    
    std::vector<float> host_input(num_elements);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-12.8f, 12.7f);  
    
    
    std::uniform_int_distribution<int> int_dist(-127, 127);
    for (int i = 0; i < num_elements; ++i) {
        host_input[i] = int_dist(gen) * scale;  
    }
    
    float *d_input, *d_dequantized;
    uint8_t *d_quantized;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_quantized, num_elements * sizeof(uint8_t));
    cudaMalloc(&d_dequantized, num_elements * sizeof(float));
    
    cudaMemcpy(d_input, host_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    std::vector<uint8_t> cpu_quantized(num_elements);
    std::vector<float> cpu_dequantized(num_elements);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_quantize(host_input.data(), cpu_quantized.data(), scale, zero_point, num_elements);
    cpu_dequantize(cpu_quantized.data(), cpu_dequantized.data(), scale, zero_point, num_elements);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    quantize_kernel<<<grid_size, block_size>>>(d_input, d_quantized, scale, zero_point, num_elements);
    dequantize_kernel<<<grid_size, block_size>>>(d_quantized, d_dequantized, scale, zero_point, num_elements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  
    
    std::vector<uint8_t> gpu_quantized(num_elements);
    std::vector<float> gpu_dequantized(num_elements);
    cudaMemcpy(gpu_quantized.data(), d_quantized, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_dequantized.data(), d_dequantized, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool quant_valid = validate_quantized(cpu_quantized.data(), gpu_quantized.data(), num_elements);
    bool dequant_valid = validate_dequantized(host_input.data(), gpu_dequantized.data(), num_elements, scale);
    
    std::cout << "Validation results:\n";
    std::cout << "Quantization: " << (quant_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "Dequantization error: " << (dequant_valid ? "WITHIN TOLERANCE" : "EXCEEDED TOLERANCE") << "\n";
    
    std::cout << "\nPerformance (quantization + dequantization):\n";
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";
    
    cudaFree(d_input);
    cudaFree(d_quantized);
    cudaFree(d_dequantized);
    return 0;
}