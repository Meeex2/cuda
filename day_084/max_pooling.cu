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
void max_pool_cpu(const float* input, float* output, 
                 int batch, int channels,
                 int in_h, int in_w,
                 int pool_h, int pool_w,
                 int stride_h, int stride_w,
                 int out_h, int out_w) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -INFINITY;
                    int h_start = oh * stride_h;
                    int w_start = ow * stride_w;
                    int h_end = h_start + pool_h;
                    int w_end = w_start + pool_w;
                    
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            if (h < in_h && w < in_w) {
                                int idx = b*channels*in_h*in_w + c*in_h*in_w + h*in_w + w;
                                max_val = fmaxf(max_val, input[idx]);
                            }
                        }
                    }
                    int out_idx = b*channels*out_h*out_w + c*out_h*out_w + oh*out_w + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

// GPU kernel
__global__ void max_pool_kernel(const float* input, float* output,
                               int in_h, int in_w,
                               int pool_h, int pool_w,
                               int stride_h, int stride_w,
                               int out_h, int out_w,
                               int channels, int batch) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (ow >= out_w || oh >= out_h || c >= channels || b >= batch) return;

    int h_start = oh * stride_h;
    int w_start = ow * stride_w;
    int h_end = h_start + pool_h;
    int w_end = w_start + pool_w;

    float max_val = -INFINITY;
    for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
            if (h < in_h && w < in_w) {
                int idx = b*channels*in_h*in_w + c*in_h*in_w + h*in_w + w;
                max_val = fmaxf(max_val, input[idx]);
            }
        }
    }
    
    int out_idx = b*channels*out_h*out_w + c*out_h*out_w + oh*out_w + ow;
    output[out_idx] = max_val;
}

// GPU implementation
void max_pool_gpu(const float* h_input, float* h_output,
                 int batch, int channels,
                 int in_h, int in_w,
                 int pool_h, int pool_w,
                 int stride_h, int stride_w,
                 int out_h, int out_w) {
    // Allocate device memory
    float *d_input, *d_output;
    size_t input_size = batch * channels * in_h * in_w * sizeof(float);
    size_t output_size = batch * channels * out_h * out_w * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

    // Configure kernel
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        batch * channels
    );

    // Launch kernel
    max_pool_kernel<<<grid, block>>>(d_input, d_output,
                                    in_h, in_w,
                                    pool_h, pool_w,
                                    stride_h, stride_w,
                                    out_h, out_w,
                                    channels, batch);

    // Copy back results
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

// Calculate output dimensions
void calc_output_size(int in_size, int pool_size, int stride, int* out_size) {
    *out_size = (in_size - pool_size) / stride + 1;
}

// Generate random image
void generate_image(float* img, int size) {
    for (int i = 0; i < size; ++i) {
        img[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

// Compare results
int compare_results(const float* a, const float* b, int size, float epsilon) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > epsilon) {
            printf("Mismatch at %d: CPU %.2f vs GPU %.2f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    // Configuration
    const int batch = 2;
    const int channels = 3;
    const int in_h = 512, in_w = 512;
    const int pool_h = 2, pool_w = 2;
    const int stride_h = 2, stride_w = 2;
    int out_h, out_w;

    calc_output_size(in_h, pool_h, stride_h, &out_h);
    calc_output_size(in_w, pool_w, stride_w, &out_w);
    
    const int input_size = batch * channels * in_h * in_w;
    const int output_size = batch * channels * out_h * out_w;
    
    // Allocate memory
    float* h_input = (float*)malloc(input_size * sizeof(float));
    float* h_output_cpu = (float*)malloc(output_size * sizeof(float));
    float* h_output_gpu = (float*)malloc(output_size * sizeof(float));

    // Generate input
    generate_image(h_input, input_size);

    // CPU implementation
    clock_t start = clock();
    max_pool_cpu(h_input, h_output_cpu, batch, channels,
                in_h, in_w, pool_h, pool_w,
                stride_h, stride_w, out_h, out_w);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // GPU implementation
    start = clock();
    max_pool_gpu(h_input, h_output_gpu, batch, channels,
                in_h, in_w, pool_h, pool_w,
                stride_h, stride_w, out_h, out_w);
    double gpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;

    // Validate
    int valid = compare_results(h_output_cpu, h_output_gpu, output_size, 1e-6f);

    printf("CPU Time: %.2fms\n", cpu_time);
    printf("GPU Time: %.2fms\n", gpu_time);
    printf("Speedup: %.1fx\n", cpu_time / gpu_time);
    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}