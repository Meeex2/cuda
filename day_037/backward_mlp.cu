#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// CUDA kernel for ReLU derivative
__global__ void relu_derivative_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0) ? 1.0f : 0.0f;
    }
}

// CUDA kernel for computing gradients of the output layer
__global__ void output_layer_gradients_kernel(const float* output, const int* labels, float* grad_output, int size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int sample_idx = idx / num_classes;
        int label = labels[sample_idx];
        grad_output[idx] = output[idx] - (idx % num_classes == label ? 1.0f : 0.0f);
    }
}


// CUDA kernel for computing gradients of the hidden layer
__global__ void hidden_layer_gradients_kernel(const float* grad_output, const float* weights, float* grad_hidden, int hidden_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int sample_idx = idx / hidden_size;
        int j = idx % hidden_size;
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            int grad_idx = sample_idx * output_size + i;
            sum += grad_output[grad_idx] * weights[j * output_size + i];
        }
        grad_hidden[idx] = sum;
    }
}

// CUDA kernel for computing gradients of weights and biases
__global__ void compute_gradients_kernel(const float* input, const float* grad_output, float* grad_weights, float* grad_biases, int hidden_size, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        // Compute gradients for weights
        for (int j = 0; j < hidden_size; j++) {
            float sum = 0.0f;
            for (int s = 0; s < batch_size; s++) {
                int input_idx = s * hidden_size + j;
                int grad_idx = s * output_size + idx;
                sum += input[input_idx] * grad_output[grad_idx];
            }
            grad_weights[j * output_size + idx] = sum;
        }

        // Compute gradient for bias
        float sum = 0.0f;
        for (int s = 0; s < batch_size; s++) {
            sum += grad_output[s * output_size + idx];
        }
        grad_biases[idx] = sum;
    }
}

// CPU reference implementation of the backward pass
void backward_pass_cpu(const std::vector<float>& output, const std::vector<int>& labels, const std::vector<float>& hidden, const std::vector<float>& weights, std::vector<float>& grad_weights, std::vector<float>& grad_biases, int hidden_size, int output_size) {
    int batch_size = labels.size();
    std::vector<float> grad_output(batch_size * output_size, 0.0f);
    for (int i = 0; i < batch_size * output_size; i++) {
        int sample_idx = i / output_size;
        int label = labels[sample_idx];
        grad_output[i] = output[i] - (i % output_size == label ? 1.0f : 0.0f);
    }

    std::vector<float> grad_hidden(batch_size * hidden_size, 0.0f);
    for (int s = 0; s < batch_size; s++) {
        for (int j = 0; j < hidden_size; j++) {
            float sum = 0.0f;
            for (int i = 0; i < output_size; i++) {
                int grad_idx = s * output_size + i;
                sum += grad_output[grad_idx] * weights[j * output_size + i];
            }
            grad_hidden[s * hidden_size + j] = sum;
        }
    }

    std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
    std::fill(grad_biases.begin(), grad_biases.end(), 0.0f);

    for (int s = 0; s < batch_size; s++) {
        for (int i = 0; i < output_size; i++) {
            int grad_idx = s * output_size + i;
            grad_biases[i] += grad_output[grad_idx];
            for (int j = 0; j < hidden_size; j++) {
                int hidden_idx = s * hidden_size + j;
                grad_weights[j * output_size + i] += hidden[hidden_idx] * grad_output[grad_idx];
            }
        }
    }
}

// Validation function
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

int main() {
    std::cout << "Starting program..." << std::endl;

    const int hidden_size = 128;
    const int output_size = 10;
    const int batch_size = 1024;
    const float learning_rate = 0.01f;

    std::vector<float> output(batch_size * output_size);
    std::vector<int> labels(batch_size);
    std::vector<float> hidden(batch_size * hidden_size);
    std::vector<float> weights(hidden_size * output_size);
    std::vector<float> grad_weights_cpu(hidden_size * output_size, 0.0f);
    std::vector<float> grad_biases_cpu(output_size, 0.0f);
    std::vector<float> grad_weights_gpu(hidden_size * output_size, 0.0f);
    std::vector<float> grad_biases_gpu(output_size, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_label(0, output_size - 1);

    for (int i = 0; i < batch_size * output_size; i++) {
        output[i] = dist(gen);
    }
    for (int i = 0; i < batch_size; i++) {
        labels[i] = dist_label(gen);
    }
    for (int i = 0; i < batch_size * hidden_size; i++) {
        hidden[i] = dist(gen);
    }
    for (int i = 0; i < hidden_size * output_size; i++) {
        weights[i] = dist(gen);
    }

    std::cout << "Initialized data with random values." << std::endl;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    backward_pass_cpu(output, labels, hidden, weights, grad_weights_cpu, grad_biases_cpu, hidden_size, output_size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    std::cout << "CPU version completed in " << cpu_duration << " seconds." << std::endl;

    float *d_output, *d_hidden, *d_weights, *d_grad_output, *d_grad_hidden, *d_grad_weights, *d_grad_biases;
    int *d_labels;

    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_labels, batch_size * sizeof(int));
    cudaMalloc(&d_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_grad_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_grad_weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_grad_biases, output_size * sizeof(float));

    cudaMemcpy(d_output, output.data(), batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden, hidden.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grad_weights, 0, hidden_size * output_size * sizeof(float));
    cudaMemset(d_grad_biases, 0, output_size * sizeof(float));

    std::cout << "Copied data to device." << std::endl;

    int block_size = 256;
    int grid_size_output = (batch_size * output_size + block_size - 1) / block_size;
    int grid_size_hidden = (batch_size * hidden_size + block_size - 1) / block_size;
    int grid_size_weights = (output_size + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    output_layer_gradients_kernel<<<grid_size_output, block_size>>>(d_output, d_labels, d_grad_output, batch_size * output_size, output_size);

    hidden_layer_gradients_kernel<<<grid_size_hidden, block_size>>>(d_grad_output, d_weights, d_grad_hidden, hidden_size, output_size, batch_size);

    compute_gradients_kernel<<<grid_size_weights, block_size>>>(d_hidden, d_grad_output, d_grad_weights, d_grad_biases, hidden_size, batch_size, output_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;

    std::cout << "GPU version completed in " << gpu_duration << " seconds." << std::endl;

    cudaMemcpy(grad_weights_gpu.data(), d_grad_weights, hidden_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_biases_gpu.data(), d_grad_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Copied GPU results back to host." << std::endl;

    bool validation_weights = validate_results(grad_weights_cpu, grad_weights_gpu, hidden_size * output_size);
    bool validation_biases = validate_results(grad_biases_cpu, grad_biases_gpu, output_size);
    std::cout << "Validation (weights): " << (validation_weights ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Validation (biases): " << (validation_biases ? "PASSED" : "FAILED") << std::endl;

    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    cudaFree(d_output);
    cudaFree(d_labels);
    cudaFree(d_hidden);
    cudaFree(d_weights);
    cudaFree(d_grad_output);
    cudaFree(d_grad_hidden);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_biases);

    std::cout << "Program completed successfully." << std::endl;

    return 0;
}

