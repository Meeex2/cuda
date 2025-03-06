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


