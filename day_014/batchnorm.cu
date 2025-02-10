#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

__global__ void batch_norm_forward(
    const float* input, 
    float* output, 
    const float* mean, 
    const float* var, 
    const float* gamma, 
    const float* beta, 
    float epsilon, 
    int batch_size, 
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_features) {
        int feature_idx = idx % num_features; 
        float normalized = (input[idx] - mean[feature_idx]) / sqrtf(var[feature_idx] + epsilon);
        output[idx] = gamma[feature_idx] * normalized + beta[feature_idx];
    }
}

void batch_norm_cpu(
    const float* input, 
    float* output, 
    const float* mean, 
    const float* var, 
    const float* gamma, 
    const float* beta, 
    float epsilon, 
    int batch_size, 
    int num_features
) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_features; j++) {
            int idx = i * num_features + j;
            float normalized = (input[idx] - mean[j]) / sqrtf(var[j] + epsilon);
            output[idx] = gamma[j] * normalized + beta[j];
        }
    }
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
int main() {
    
    const int batch_size = 128;
    const int num_features = 64;
    const int num_elements = batch_size * num_features;
    const float epsilon = 1e-5;
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    
    float* h_input = new float[num_elements];
    float* h_output_cpu = new float[num_elements];
    float* h_output_gpu = new float[num_elements];
    float* h_mean = new float[num_features];
    float* h_var = new float[num_features];
    float* h_gamma = new float[num_features];
    float* h_beta = new float[num_features];
    
    for (int i = 0; i < num_elements; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
    for (int j = 0; j < num_features; j++) {
        h_mean[j] = 0.5f;
        h_var[j] = 1.0f;
        h_gamma[j] = 1.0f;
        h_beta[j] = 0.0f;
    }
    
    float *d_input, *d_output, *d_mean, *d_var, *d_gamma, *d_beta;
    CHECK_CUDA(cudaMalloc(&d_input, num_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, num_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mean, num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_var, num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gamma, num_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, num_features * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, num_elements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mean, h_mean, num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_var, h_var, num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta, num_features * sizeof(float), cudaMemcpyHostToDevice));
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    batch_norm_cpu(h_input, h_output_cpu, h_mean, h_var, h_gamma, h_beta, epsilon, batch_size, num_features);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    batch_norm_forward<<<blocks_per_grid, threads_per_block>>>(
        d_input, d_output, d_mean, d_var, d_gamma, d_beta, epsilon, batch_size, num_features
    );
    CHECK_CUDA(cudaDeviceSynchronize()); 
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;
    
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < num_elements; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "Test passed! CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Test failed! CPU and GPU results do not match." << std::endl;
    }
    
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    delete[] h_mean;
    delete[] h_var;
    delete[] h_gamma;
    delete[] h_beta;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_var));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));
    return 0;
}