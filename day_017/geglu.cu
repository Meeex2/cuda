#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

__global__ void geglu_kernel(const float* input, float* output, int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements / 2) return;
    const int half_dim = num_elements / 2;
    const float x = input[idx];
    const float y = input[idx + half_dim];
    
    
    const float gelu = x * 0.5f * (1.0f + tanhf(1.702f * x));
    
    output[idx] = gelu * y;
}

void geglu_cpu(const float* input, float* output, int num_elements) {
    const int half_dim = num_elements / 2;
    for (int i = 0; i < half_dim; ++i) {
        const float x = input[i];
        const float y = input[i + half_dim];
        
        
        const float gelu = x * 0.5f * (1.0f + std::tanh(1.702f * x));
        
        output[i] = gelu * y;
    }
}

void test_geglu() {
    const int num_elements = 1 << 20; 
    const int half_dim = num_elements / 2;
    std::vector<float> h_input(num_elements);
    std::vector<float> h_output_cpu(half_dim);
    std::vector<float> h_output_gpu(half_dim);
    
    std::default_random_engine engine;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_input) v = dist(engine);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_output, half_dim * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    geglu_cpu(h_input.data(), h_output_cpu.data(), num_elements);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    
    const int block_size = 256;
    const int grid_size = (half_dim + block_size - 1) / block_size;
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    geglu_kernel<<<grid_size, block_size>>>(d_input, d_output, num_elements);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    
    cudaMemcpy(h_output_gpu.data(), d_output, half_dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_error = 0.0f;
    for (int i = 0; i < half_dim; ++i) {
        float error = fabs(h_output_cpu[i] - h_output_gpu[i]);
        if (error > max_error) max_error = error;
    }
    std::cout << "Validation Results:\n";
    std::cout << "Max absolute error: " << max_error << "\n";
    std::cout << "\nTiming Results:\n";
    std::cout << "CPU Time: " << cpu_duration.count() << " s\n";
    std::cout << "GPU Time: " << gpu_duration.count() << " s\n";
    std::cout << "Speedup: " << cpu_duration.count() / gpu_duration.count() << "x\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}
int main() {
    test_geglu();
    return 0;
}
