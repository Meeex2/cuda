#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

__global__ void smooth_swiglu_kernel(const float* input, float* output, int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float x = input[i];
        const float sigmoid = 1.0f / (1.0f + __expf(-x));  
        const float swish = x * sigmoid;                  
        const float glu = x * swish;                      
        output[i] = glu / (1.0f + __expf(-glu));          
    }
}

void cpu_smooth_swiglu(const float* input, float* output, int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        const float x = input[i];
        const float sigmoid = 1.0f / (1.0f + std::exp(-x));  
        const float swish = x * sigmoid;                     
        const float glu = x * swish;                         
        output[i] = glu / (1.0f + std::exp(-glu));           
    }
}



