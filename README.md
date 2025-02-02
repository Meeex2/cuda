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
**File:** `ReLU.cu`

### Summary
Implemented the ReLU (Rectified Linear Unit) function using CUDA.

Wrote a CUDA kernel to perform an element-wise ReLU operation on an array, where each thread computes the ReLU of a single element.


## Day 3 Challenge

**File:** `scalar_product.cu`


### Summary
Implemented the scalar product (dot product) of two vectors using CUDA. Explored how to launch a kernel to perform parallelized multiplication and reduction, where each thread computes the product of a pair of values, and the results are summed up to produce the final scalar product.


### Learned
- **Basics of Writing a CUDA Kernel:** Understood the structure and syntax of a CUDA kernel function.
- **Grid, Block, and Thread Hierarchy:** Explored the hierarchical organization of threads, blocks, and grids in CUDA.
- **Device Memory Management:**
  - Allocated device memory using `cudaMalloc`.
  - Copied data between host and device using `cudaMemcpy`.
  - Freed device memory using `cudaFree`.
- **Shared Memory Usage:** Utilized shared memory for efficient reduction within a block.
- **Reduction Algorithm:** Implemented a parallel reduction to sum partial results computed by threads.
- **Host Array Initialization:** Initialized arrays on the host and copied them to the device.
- **Kernel Launch Configuration:** Configured the number of threads per block and the number of blocks per grid for kernel launch.


## Day 4 Challenge

**File:** `MatVec_prod.cu`


### **Summary**
Implemented the matrix-vector product using CUDA. Explored how to launch a kernel to perform parallelized matrix-vector multiplication, where each thread computes the dot product of a row of the matrix with the vector. This builds on the concepts learned in previous challenges but extends them to handle matrix operations.

**File:** `LeakyReLU.cu`


### **Summary**
Implemented the most basic LeakyReLU function (linear leak) using CUDA.


## Day 5 Challenge

**File:** `Mat_prod.cu`


### **Summary**
Implemented the matrix multiplication using CUDA. Explored how to launch a kernel to perform parallelized matrix-matrix multiplication use x and y axis of the grid and the blcoks, where each thread computes a coeffecient of the result matrix.


## Day 6 Challenge

**File:** `imageBlur.cu`


### **Summary**
Implemented the matrix multiplication using CUDA. Explored how to launch a kernel to perform parallelized matrix-matrix multiplication use x and y axis of the grid and the blcoks, where each thread computes a coeffecient of the result matrix.


### **Day 5 Challenge: Image Blurring with CUDA**

**File:** `image_blur.cu`


### **Summary**
Implemented image blurring using CUDA. Explored how to perform a **convolution operation** on an image using a Gaussian blur kernel. Each thread computes one pixel of the output image by applying the kernel to the corresponding neighborhood in the input image.


### **Key Concepts**
1. **Basics of Image Convolution**:
   - Understood how convolution works for image processing, where each pixel in the output image is a weighted sum of its neighborhood in the input image.
   - Applied a **Gaussian blur kernel** to smooth the image.

2. **CUDA Kernel for Image Processing**:
   - Wrote a CUDA kernel to parallelize the convolution operation.
   - Each thread computes one pixel of the output image, ensuring efficient use of GPU resources.

3. **Handling Edge Cases**:
   - Implemented **clamping** to handle edge pixels, ensuring the kernel does not access out-of-bounds memory.

4. **Grid and Block Configuration**:
   - Configured the number of threads per block and the number of blocks per grid to cover the entire image.
   - Used 2D thread blocks and grids to match the 2D nature of images.

5. **Verification and Debugging**:
   - Verified the correctness of the output by manually computing the expected value of a pixel and comparing it with the GPU result.
   - Printed a small portion of the output image for visual inspection.


