#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
namespace cg = cooperative_groups;

__device__ __inline__ float fast_swish(float x) {
    
    const float beta = 1.702f;
    float x_b = beta * x;
    return x * (0.5f + x_b * (0.148f + x_b * (0.009f + x_b * 0.003f))) / 
           (1.0f + fabsf(x_b * (0.236f + x_b * (0.087f + x_b * 0.014f))));
}
__global__ void swiglu_kernel(const float* __restrict__ input, 
                             float* __restrict__ output, 
                             int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int half_dim = num_elements / 2;
    
    for(int i = tid * 4; i < half_dim; i += stride * 4) {
        float4 in_vec, gate_vec;
        int load_pos = i/4;
        
        
        if(load_pos < (half_dim/4)) {
            in_vec = reinterpret_cast<const float4*>(input)[load_pos];
            gate_vec = reinterpret_cast<const float4*>(input + half_dim)[load_pos];
        } else {
            
            in_vec = {}; gate_vec = {};
            float* in_ptr = reinterpret_cast<float*>(&in_vec);
            float* gate_ptr = reinterpret_cast<float*>(&gate_vec);
            for(int j=0; j<4 && (i+j)<half_dim; j++) {
                in_ptr[j] = input[i+j];
                gate_ptr[j] = input[half_dim + i+j];
            }
        }
        float results[4];
        #pragma unroll
        for(int j=0; j<4; j++) {
            float x = reinterpret_cast<float*>(&in_vec)[j];
            float g = reinterpret_cast<float*>(&gate_vec)[j];
            
            
            float swish = fast_swish(x);
            results[j] = swish * g;
        }
        
        if(load_pos < (half_dim/4)) {
            reinterpret_cast<float4*>(output)[load_pos] = *reinterpret_cast<float4*>(results);
        } else {
            for(int j=0; j<4 && (i+j)<half_dim; j++) {
                output[i+j] = results[j];
            }
        }
    }
}

void swiglu_cpu(const float* input, float* output, int num_elements) {
    const int half_dim = num_elements / 2;
    for(int i=0; i<half_dim; i++) {
        float x = input[i];
        float g = input[i + half_dim];
        float swish = x / (1.0f + expf(-1.702f * x));
        output[i] = swish * g;
    }
}


void test_swiglu() {
    const int num_elements = 1 << 24;  
    std::vector<float> h_input(num_elements);
    std::vector<float> h_output_cpu(num_elements/2);
    std::vector<float> h_output_gpu(num_elements/2);
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for(auto& v : h_input) v = dist(gen);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_output, (num_elements/2) * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    swiglu_cpu(h_input.data(), h_output_cpu.data(), num_elements);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaEventRecord(start);
    const int block_size = 256;
    const int grid_size = (num_elements/8 + block_size - 1) / block_size;
    swiglu_kernel<<<grid_size, block_size>>>(d_input, d_output, num_elements);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, end);
    gpu_time /= 1000.0f;
    
    cudaMemcpy(h_output_gpu.data(), d_output, (num_elements/2)*sizeof(float), cudaMemcpyDeviceToHost);
    
    double mse = 0.0f;
    float max_err = 0.0f;
    for(int i=0; i<num_elements/2; i++) {
        float err = fabs(h_output_cpu[i] - h_output_gpu[i]);
        mse += err * err;
        max_err = fmaxf(max_err, err);
    }
    mse /= (num_elements/2);
    std::cout << "Validation Results:\n"
              << "MSE: " << mse << "\n"
              << "Max Error: " << max_err << "\n\n"
              << "Timing Results:\n"
              << "CPU Time: " << cpu_time << " s\n"
              << "GPU Time: " << gpu_time << " s\n"
              << "Speedup: " << cpu_time/gpu_time << "x\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}
int main() {
    test_swiglu();
    return 0;
}