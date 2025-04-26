#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// CPU implementation
void conv1d_cpu(const float* input, const float* kernel, float* output,
                int input_size, int kernel_size,
                int padding, int stride) {
    int output_size = (input_size + 2*padding - kernel_size) / stride + 1;
    
    for (int o = 0; o < output_size; ++o) {
        float sum = 0.0f;
        int input_start = o * stride - padding;
        
        for (int k = 0; k < kernel_size; ++k) {
            int i = input_start + k;
            if (i >= 0 && i < input_size) {
                sum += input[i] * kernel[k];
            }
        }
        output[o] = sum;
    }
}

// GPU kernel
__global__ void conv1d_kernel(const float* input, const float* kernel, float* output,
                             int input_size, int kernel_size,
                             int padding, int stride,
                             int output_size) {
    extern __shared__ float s_kernel[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load kernel into shared memory
    for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
        if (i < kernel_size) s_kernel[i] = kernel[i];
    }
    __syncthreads();

    if (tid >= output_size) return;

    float sum = 0.0f;
    int input_start = tid * stride - padding;
    
    for (int k = 0; k < kernel_size; ++k) {
        int input_idx = input_start + k;
        if (input_idx >= 0 && input_idx < input_size) {
            sum += input[input_idx] * s_kernel[k];
        }
    }
    output[tid] = sum;
}

// GPU implementation
void conv1d_gpu(const float* h_input, const float* h_kernel, float* h_output,
               int input_size, int kernel_size,
               int padding, int stride) {
    int output_size = (input_size + 2*padding - kernel_size) / stride + 1;
    
    float *d_input, *d_kernel, *d_output;
    size_t input_bytes = input_size * sizeof(float);
    size_t kernel_bytes = kernel_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;
    size_t shared_mem = kernel_size * sizeof(float);

    conv1d_kernel<<<grid_size, block_size, shared_mem>>>(
        d_input, d_kernel, d_output,
        input_size, kernel_size,
        padding, stride,
        output_size
    );

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

// Generate random signal
void generate_signal(float* signal, int size) {
    for (int i = 0; i < size; ++i) {
        signal[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    }
}

// Compare results
int compare_results(const float* cpu, const float* gpu, int size, float epsilon) {
    for (int i = 0; i < size; ++i) {
        if (fabs(cpu[i] - gpu[i]) > epsilon) {
            printf("Mismatch at %d: CPU %.4f vs GPU %.4f\n", i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    // Configuration
    const int input_size = 1 << 18;  // 262,144 elements
    const int kernel_size = 128;
    const int padding = 16;
    const int stride = 2;
    
    // Calculate output size
    const int output_size = (input_size + 2*padding - kernel_size) / stride + 1;

    // Allocate memory
    float* h_input = (float*)malloc(input_size * sizeof(float));
    float* h_kernel = (float*)malloc(kernel_size * sizeof(float));
    float* h_output_cpu = (float*)malloc(output_size * sizeof(float));
    float* h_output_gpu = (float*)malloc(output_size * sizeof(float));

    // Generate random input and kernel
    generate_signal(h_input, input_size);
    generate_signal(h_kernel, kernel_size);

    // CPU computation
    clock_t start = clock();
    conv1d_cpu(h_input, h_kernel, h_output_cpu,
              input_size, kernel_size,
              padding, stride);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // GPU computation
    start = clock();
    conv1d_gpu(h_input, h_kernel, h_output_gpu,
              input_size, kernel_size,
              padding, stride);
    double gpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // Validation
    int valid = compare_results(h_output_cpu, h_output_gpu, output_size, 1e-4f);

    printf("Input size: %d\n", input_size);
    printf("Kernel size: %d\n", kernel_size);
    printf("Output size: %d\n", output_size);
    printf("CPU time: %.2fms\n", cpu_time);
    printf("GPU time: %.2fms\n", gpu_time);
    printf("Speedup: %.1fx\n", cpu_time / gpu_time);
    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

    // Cleanup
    free(h_input);
    free(h_kernel);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}