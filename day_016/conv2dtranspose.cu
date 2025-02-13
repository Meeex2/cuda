
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int H_in, int W_in, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * H_out * W_out;
    if (idx >= total_output) return;
    
    int n = idx / (out_channels * H_out * W_out);
    int c_out = (idx / (H_out * W_out)) % out_channels;
    int oh = (idx / W_out) % H_out;
    int ow = idx % W_out;
    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            
            int ih = (oh - kh + padding_h) / stride_h;
            int iw = (ow - kw + padding_w) / stride_w;
            
            
            if ((oh - kh + padding_h) % stride_h != 0) continue;
            if ((ow - kw + padding_w) % stride_w != 0) continue;
            if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
            
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                int input_idx = n * (in_channels * H_in * W_in) 
                              + c_in * (H_in * W_in) 
                              + ih * W_in + iw;
                int weight_idx = c_in * (out_channels * kernel_h * kernel_w)
                               + c_out * (kernel_h * kernel_w)
                               + kh * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    output[idx] = sum;
}

void conv_transpose2d_cpu(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int H_in, int W_in, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    for (int n = 0; n < batch_size; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = (oh - kh + padding_h) / stride_h;
                            int iw = (ow - kw + padding_w) / stride_w;
                            
                            if ((oh - kh + padding_h) % stride_h != 0) continue;
                            if ((ow - kw + padding_w) % stride_w != 0) continue;
                            if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
                            for (int c_in = 0; c_in < in_channels; ++c_in) {
                                int input_idx = n * (in_channels * H_in * W_in)
                                              + c_in * (H_in * W_in)
                                              + ih * W_in + iw;
                                int weight_idx = c_in * (out_channels * kernel_h * kernel_w)
                                               + c_out * (kernel_h * kernel_w)
                                               + kh * kernel_w + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                    int output_idx = n * (out_channels * H_out * W_out)
                                   + c_out * (H_out * W_out)
                                   + oh * W_out + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

void test_conv_transpose2d() {
    
    int batch_size = 1;
    int in_channels = 1;
    int out_channels = 1;
    int H_in = 3, W_in = 3;
    int kernel_h = 3, kernel_w = 3;
    int stride_h = 2, stride_w = 2;
    int padding_h = 1, padding_w = 1;
    int H_out = (H_in - 1) * stride_h + kernel_h - 2 * padding_h;
    int W_out = (W_in - 1) * stride_w + kernel_w - 2 * padding_w;
    
    std::vector<float> h_input(batch_size * in_channels * H_in * W_in, 1.0f);
    std::vector<float> h_weight(out_channels * in_channels * kernel_h * kernel_w, 1.0f);
    std::vector<float> h_output_cpu(batch_size * out_channels * H_out * W_out);
    std::vector<float> h_output_gpu(h_output_cpu.size());
    
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
    cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    conv_transpose2d_cpu(
        h_input.data(), h_weight.data(), h_output_cpu.data(),
        batch_size, in_channels, out_channels,
        H_in, W_in, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w
    );
    
    int total_output = batch_size * out_channels * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;
    
    conv_transpose2d_kernel<<<grid_size, block_size>>>(
        d_input, d_weight, d_output,
        batch_size, in_channels, out_channels,
        H_in, W_in, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w
    );
    cudaMemcpy(h_output_gpu.data(), d_output, h_output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    float tolerance = 1e-4;
    for (size_t i = 0; i < h_output_cpu.size(); ++i) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i 
                      << ": CPU=" << h_output_cpu[i] 
                      << ", GPU=" << h_output_gpu[i] << std::endl;
            return;
        }
    }
    std::cout << "Test passed!" << std::endl;
    
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
int main() {
    test_conv_transpose2d();
    return 0;
}