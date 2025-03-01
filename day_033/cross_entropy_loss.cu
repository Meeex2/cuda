#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// CUDA kernel for cross-entropy loss
__global__ void cross_entropy_loss_kernel(const float* predictions, const int* labels, float* loss, int num_samples, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        int label = labels[idx];
        float prob = predictions[idx * num_classes + label];
        loss[idx] = -logf(prob + 1e-9);  // Add small epsilon to avoid log(0)
    }
}

// CPU reference implementation of cross-entropy loss
float cross_entropy_loss_cpu(const std::vector<float>& predictions, const std::vector<int>& labels, int num_samples, int num_classes) {
    float loss = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        int label = labels[i];
        float prob = predictions[i * num_classes + label];
        loss += -logf(prob + 1e-9);  // Add small epsilon to avoid log(0)
    }
    return loss / num_samples;
}

// Validation function
bool validate_results(float cpu_loss, float gpu_loss, float tolerance = 1e-3) {
    if (std::fabs(cpu_loss - gpu_loss) > tolerance) {
        std::cout << "Mismatch: CPU=" << cpu_loss << ", GPU=" << gpu_loss << std::endl;
        return false;
    }
    return true;
}

int main() {
    // Problem dimensions
    const int num_samples = 100000;  
    const int num_classes = 10;

    // Allocate host memory
    std::vector<float> predictions(num_samples * num_classes);
    std::vector<int> labels(num_samples);
    std::vector<float> loss(num_samples, 0.0f);

    // Initialize predictions and labels with random values
    std::mt19937 gen(42);  // Random number generator
    std::uniform_real_distribution<float> dist_pred(0.0f, 1.0f);  // Uniform distribution for predictions
    std::uniform_int_distribution<int> dist_label(0, num_classes - 1);  // Uniform distribution for labels

    for (int i = 0; i < num_samples; i++) {
        labels[i] = dist_label(gen);  // Random label
        for (int j = 0; j < num_classes; j++) {
            predictions[i * num_classes + j] = dist_pred(gen);  // Random prediction
        }
        // Normalize predictions to make them probabilities
        float sum = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum += predictions[i * num_classes + j];
        }
        for (int j = 0; j < num_classes; j++) {
            predictions[i * num_classes + j] /= sum;
        }
    }

    // Run CPU version
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_loss = cross_entropy_loss_cpu(predictions, labels, num_samples, num_classes);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    // Allocate device memory
    float *d_predictions, *d_loss;
    int *d_labels;

    cudaMalloc(&d_predictions, num_samples * num_classes * sizeof(float));
    cudaMalloc(&d_labels, num_samples * sizeof(int));
    cudaMalloc(&d_loss, num_samples * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_predictions, predictions.data(), num_samples * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;

    // Run GPU version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cross_entropy_loss_kernel<<<grid_size, block_size>>>(d_predictions, d_labels, d_loss, num_samples, num_classes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  // Convert to seconds

    // Copy GPU results back
    cudaMemcpy(loss.data(), d_loss, num_samples * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute average loss on GPU
    float gpu_loss = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        gpu_loss += loss[i];
    }
    gpu_loss /= num_samples;

    // Validate results
    bool validation = validate_results(cpu_loss, gpu_loss);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;

    // Print timings
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    // Cleanup
    cudaFree(d_predictions);
    cudaFree(d_labels);
    cudaFree(d_loss);

    return 0;
}