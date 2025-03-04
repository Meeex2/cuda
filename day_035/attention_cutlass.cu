#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

__global__ void softmax_kernel(const float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float max_val = input[row * num_cols];
        for (int col = 1; col < num_cols; col++) {
            max_val = fmaxf(max_val, input[row * num_cols + col]);
        }
        float sum = 0.0f;
        for (int col = 0; col < num_cols; col++) {
            output[row * num_cols + col] = expf(input[row * num_cols + col] - max_val);
            sum += output[row * num_cols + col];
        }
        for (int col = 0; col < num_cols; col++) {
            output[row * num_cols + col] /= sum;
        }
    }
}

__global__ void weighted_aggregation_kernel(const float* attention_weights, const float* values, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (int col = 0; col < num_cols; col++) {
            float sum = 0.0f;
            for (int k = 0; k < num_cols; k++) {
                sum += attention_weights[row * num_cols + k] * values[k * num_cols + col];
            }
            output[row * num_cols + col] = sum;
        }
    }
}

void self_attention_cpu(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& output, int num_rows, int num_cols) {
    std::vector<float> scores(num_rows * num_cols, 0.0f);
    std::vector<float> attention_weights(num_rows * num_cols, 0.0f);
    
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_cols; k++) {
                sum += Q[i * num_cols + k] * K[j * num_cols + k];
            }
            scores[i * num_cols + j] = sum / sqrtf(num_cols);
        }
    }
    
    for (int i = 0; i < num_rows; i++) {
        float max_val = scores[i * num_cols];
        for (int j = 1; j < num_cols; j++) {
            max_val = fmaxf(max_val, scores[i * num_cols + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < num_cols; j++) {
            attention_weights[i * num_cols + j] = expf(scores[i * num_cols + j] - max_val);
            sum += attention_weights[i * num_cols + j];
        }
        for (int j = 0; j < num_cols; j++) {
            attention_weights[i * num_cols + j] /= sum;
        }
    }
    
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < num_cols; k++) {
                sum += attention_weights[i * num_cols + k] * V[k * num_cols + j];
            }
            output[i * num_cols + j] = sum;
        }
    }
}

bool validate_results(const std::vector<float>& cpu, const std::vector<float>& gpu, int size, float tolerance = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(cpu[i] - gpu[i]) > tolerance) {
            std::cout << "Mismatch at index " << i 
                      << ": CPU=" << cpu[i] 
                      << ", GPU=" << gpu[i] 
                      << std::endl;
            return false;
        }
    }
    return true;
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
    std::cout << "Starting program..." << std::endl;
    
    const int num_rows = 128;
    const int num_cols = 128;
    
    std::vector<float> Q(num_rows * num_cols);
    std::vector<float> K(num_rows * num_cols);
    std::vector<float> V(num_rows * num_cols);
    std::vector<float> output_cpu(num_rows * num_cols, 0.0f);
    std::vector<float> output_gpu(num_rows * num_cols, 0.0f);
    
    std::mt19937 gen(42);  
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);  
    for (int i = 0; i < num_rows * num_cols; i++) {
        Q[i] = dist(gen);
        K[i] = dist(gen);
        V[i] = dist(gen);
    }
    std::cout << "Initialized Q, K, V with random values." << std::endl;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    self_attention_cpu(Q, K, V, output_cpu, num_rows, num_cols);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();
    std::cout << "CPU version completed in " << cpu_duration << " seconds." << std::endl;
    
    float *d_Q, *d_K, *d_V, *d_scores, *d_attention_weights, *d_output;
    CHECK_CUDA(cudaMalloc(&d_Q, num_rows * num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, num_rows * num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, num_rows * num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, num_rows * num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attention_weights, num_rows * num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, num_rows * num_cols * sizeof(float)));
    std::cout << "Allocated device memory." << std::endl;
    
    CHECK_CUDA(cudaMemcpy(d_Q, Q.data(), num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, K.data(), num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, V.data(), num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "Copied data to device." << std::endl;
    
    using Gemm = cutlass::gemm::device::Gemm<
        float,                                   
        cutlass::layout::RowMajor,              
        float,                                   
        cutlass::layout::ColumnMajor,           
        float,                                   
        cutlass::layout::RowMajor,              
        float,                                   
        cutlass::arch::OpClassSimt,             
        cutlass::arch::Sm75                     
    >;
    
    typename Gemm::Arguments args(
        {num_rows, num_cols, num_cols},         
        {d_Q, num_cols},                        
        {d_K, num_cols},                        
        {d_scores, num_cols},                   
        {d_scores, num_cols},                   
        {1.0f / sqrtf(num_cols), 0.0f},         
        0                                       
    );
    
    Gemm gemm_op;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    cutlass::Status status = gemm_op(args);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float gpu_duration;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_duration, start, stop)));
    gpu_duration /= 1000.0f;  
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed with error: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }
    std::cout << "CUTLASS GEMM completed in " << gpu_duration << " seconds." << std::endl;
    
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    softmax_kernel<<<grid_size, block_size>>>(d_scores, d_attention_weights, num_rows, num_cols);
    CHECK_CUDA(cudaGetLastError());
    std::cout << "Softmax kernel executed." << std::endl;
    
    weighted_aggregation_kernel<<<grid_size, block_size>>>(d_attention_weights, d_V, d_output, num_rows, num_cols);
    CHECK_CUDA(cudaGetLastError());
    std::cout << "Weighted aggregation kernel executed." << std::endl;
    
    CHECK_CUDA(cudaMemcpy(output_gpu.data(), d_output, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Copied GPU results back to host." << std::endl;
    
    bool validation = validate_results(output_cpu, output_gpu, num_rows * num_cols);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";
    
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_attention_weights));
    CHECK_CUDA(cudaFree(d_output));

    std::cout << "Program completed successfully." << std::endl;

    return 0;
}
    

