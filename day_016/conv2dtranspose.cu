
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

