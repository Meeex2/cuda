#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// CUDA kernel for computing perplexity
__global__ void perplexity_kernel(const float* probabilities, const int* labels, float* log_loss, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        int label = labels[idx];
        float prob = probabilities[idx * num_samples + label];  // Assuming num_classes == num_samples
        log_loss[idx] = -logf(prob + 1e-9);  // Add small epsilon to avoid log(0)
    }
}

// CPU reference implementation of perplexity
float perplexity_cpu(const std::vector<float>& probabilities, const std::vector<int>& labels, int num_samples) {
    float log_loss_sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        int label = labels[i];
        float prob = probabilities[i * num_samples + label];  // Assuming num_classes == num_samples
        log_loss_sum += -logf(prob + 1e-9);  // Add small epsilon to avoid log(0)
    }
    return expf(log_loss_sum / num_samples);  // Compute perplexity
}

// Validation function
bool validate_results(float cpu_perplexity, float gpu_perplexity, float tolerance = 1e-3) {
    if (std::fabs(cpu_perplexity - gpu_perplexity) > tolerance) {
        std::cout << "Mismatch: CPU=" << cpu_perplexity << ", GPU=" << gpu_perplexity << std::endl;
        return false;
    }
    return true;
}

int main() {
    // Problem dimensions
    const int num_samples = 10000;  // Number of samples
    const int num_classes = num_samples;  // Assuming num_classes == num_samples for simplicity

    // Allocate host memory
    size_t total_elements = static_cast<size_t>(num_samples) * static_cast<size_t>(num_classes);
    std::vector<float> probabilities(total_elements);
    std::vector<int> labels(num_samples);
    std::vector<float> log_loss(num_samples, 0.0f);

    // Initialize probabilities and labels with random values
    std::mt19937 gen(42);  // Random number generator
    std::uniform_real_distribution<float> dist_prob(0.0f, 1.0f);  // Uniform distribution for probabilities
    std::uniform_int_distribution<int> dist_label(0, num_classes - 1);  // Uniform distribution for labels

    for (int i = 0; i < num_samples; i++) {
        labels[i] = dist_label(gen);  // Random label
        for (int j = 0; j < num_classes; j++) {
            probabilities[i * num_classes + j] = dist_prob(gen);  // Random probability
        }
        // Normalize probabilities to make them valid
        float sum = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum += probabilities[i * num_classes + j];
        }
        for (int j = 0; j < num_classes; j++) {
            probabilities[i * num_classes + j] /= sum;
        }
    }

    // Run CPU version
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_perplexity = perplexity_cpu(probabilities, labels, num_samples);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_duration = std::chrono::duration<float>(cpu_end - cpu_start).count();

    // Allocate device memory
    float *d_probabilities, *d_log_loss;
    int *d_labels;

    size_t probabilities_size = total_elements * sizeof(float);
    size_t labels_size = num_samples * sizeof(int);
    size_t log_loss_size = num_samples * sizeof(float);

    cudaMalloc(&d_probabilities, probabilities_size);
    cudaMalloc(&d_labels, labels_size);
    cudaMalloc(&d_log_loss, log_loss_size);

    // Copy data to device
    cudaMemcpy(d_probabilities, probabilities.data(), probabilities_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), labels_size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;

    // Run GPU version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    perplexity_kernel<<<grid_size, block_size>>>(d_probabilities, d_labels, d_log_loss, num_samples);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);
    gpu_duration /= 1000.0f;  // Convert to seconds

    // Copy GPU results back
    cudaMemcpy(log_loss.data(), d_log_loss, log_loss_size, cudaMemcpyDeviceToHost);

    // Compute average log loss and perplexity on GPU
    float gpu_log_loss_sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        gpu_log_loss_sum += log_loss[i];
    }
    float gpu_perplexity = expf(gpu_log_loss_sum / num_samples);

    // Validate results
    bool validation = validate_results(cpu_perplexity, gpu_perplexity);
    std::cout << "Validation: " << (validation ? "PASSED" : "FAILED") << std::endl;

    // Print timings
    std::cout << "CPU time: " << cpu_duration << " seconds\n";
    std::cout << "GPU time: " << gpu_duration << " seconds\n";
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x\n";

    // Cleanup
    cudaFree(d_probabilities);
    cudaFree(d_labels);
    cudaFree(d_log_loss);

    return 0;
}