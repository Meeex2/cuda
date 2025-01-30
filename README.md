# CUDA 100 days challenge
100 days of building Cuda kernels!

This document serves as documentaion of my learning journey of CUDA programming and studying the PMPP (Parallel Programming and Optimization) book.

## Mentor
[hkproj](https://github.com/hkproj/)

## Day 1
**File:** `elementwise_mult.cu`

### Summary
Implemented array multiplication by writing a CUDA program. Explored how to launch a kernel to perform a parallelized multiplication of two arrays, where each thread computes the product of a pair of values.

### Learned
- **Basics of Writing a CUDA Kernel:** Understood the structure and syntax of a CUDA kernel function.
- **Grid, Block, and Thread Hierarchy:** Gained insight into the hierarchical organization of threads, blocks, and grids in CUDA.
- **Device Memory Management:**
  - Allocated device memory using `cudaMalloc`.
  - Copied data between host and device using `cudaMemcpy`.
  - Freed device memory using `cudaFree`.
- **Host Array Initialization:** Initialized arrays on the host and copied them to the device.
- **Kernel Launch Configuration:** Configured the number of threads per block and the number of blocks per grid for kernel launch.

## Day 2
**File:** `relu.cu`

### Summary
Implemented the ReLU (Rectified Linear Unit) function using CUDA.
Wrote a CUDA kernel to perform an element-wise ReLU operation on an array, where each thread computes the ReLU of a single element.
