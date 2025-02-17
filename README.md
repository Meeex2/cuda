# CUDA 100 days challenge
100 days of building Cuda kernels!

This document serves as documentation of my learning journey of CUDA programming and studying the PMPP (Parallel Programming and Optimization) book.

## Mentor
[hkproj](https://github.com/hkproj/)

---

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

---

## Day 2
**File:** `ReLU.cu`

### Summary
Implemented the ReLU (Rectified Linear Unit) function using CUDA. Wrote a CUDA kernel to perform an element-wise ReLU operation on an array, where each thread computes the ReLU of a single element.

---

## Day 3
**File:** `scalar_product.cu`

### Summary
Implemented the scalar product (dot product) of two vectors using CUDA. Explored how to launch a kernel to perform parallelized multiplication and reduction, where each thread computes the product of a pair of values, and the results are summed up to produce the final scalar product.

### Learned
- **Shared Memory Usage:** Utilized shared memory for efficient reduction within a block.
- **Reduction Algorithm:** Implemented a parallel reduction to sum partial results computed by threads.

---

## Day 4
**File:** `MatVec_prod.cu`

### Summary
Implemented the matrix-vector product using CUDA. Explored how to launch a kernel to perform parallelized matrix-vector multiplication, where each thread computes the dot product of a row of the matrix with the vector. This builds on the concepts learned in previous challenges but extends them to handle matrix operations.

**File:** `LeakyReLU.cu`

### Summary
Implemented the most basic LeakyReLU function (linear leak) using CUDA.

---

## Day 5
**File:** `Mat_prod.cu`

### Summary
Implemented matrix multiplication using CUDA. Explored how to launch a kernel to perform parallelized matrix-matrix multiplication using the x and y axes of the grid and blocks, where each thread computes a coefficient of the result matrix.

---

## Day 6
**File:** `image_blur.cu`

### Summary
Implemented image blurring using CUDA. Explored how to perform a **convolution operation** on an image using a Gaussian blur kernel. Each thread computes one pixel of the output image by applying the kernel to the corresponding neighborhood in the input image.

### Key Concepts
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

---

## Day 7
**File:** `max_pooling.cu`

### Summary
Implemented max pooling using CUDA. Explored how to perform a **downsampling operation** on an input feature map using a pooling window. Each thread computes the maximum value in its corresponding window of the input feature map and stores it in the output feature map.

### Key Concepts
1. **Basics of Max Pooling**:
   - Understood how max pooling works, where each output pixel is the maximum value in a fixed-size window of the input feature map.
   - Applied a **2x2 pooling window** to downsample the input feature map.

2. **CUDA Kernel for Max Pooling**:
   - Wrote a CUDA kernel to parallelize the max pooling operation.
   - Each thread computes the maximum value in its assigned window, ensuring efficient use of GPU resources.

3. **Handling Edge Cases**:
   - Implemented **clamping** to handle edge pixels, ensuring the kernel does not access out-of-bounds memory.

4. **Grid and Block Configuration**:
   - Configured the number of threads per block and the number of blocks per grid to cover the entire output feature map.
   - Used 2D thread blocks and grids to match the 2D nature of the feature map.

5. **Verification and Debugging**:
   - Verified the correctness of the output by manually computing the expected maximum value for a specific window and comparing it with the GPU result.
   - Printed a small portion of the output feature map for visual inspection.

## Day 8
**Files:** `softmax.cu`, `softmax_v2.cu`

### **Summary**
Implemented the **softmax function** using CUDA. The softmax function is widely used in machine learning to convert a vector of raw scores (logits) into probabilities. Two versions were implemented:
1. **Basic Softmax**: A straightforward implementation using global memory.
2. **Advanced Softmax**: An optimized implementation using **shared memory** for efficient reduction and normalization.

### **Key Concepts**
1. **Softmax Formula**:
   - The softmax probability for the $i$-th element is given by:
     
     $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$
     
   - This involves computing the exponential of each element and normalizing by the sum of all exponentials.

2. **Basic Softmax**:
   - Each thread computes the exponential of one element of the input vector.
   - The sum of exponentials is computed on the host, and a second kernel normalizes the values.

3. **Advanced Softmax**:
   - Uses **shared memory** to perform parallel reduction for computing the sum of exponentials.
   - The reduction kernel efficiently sums the exponentials in parallel, reducing global memory access.
   - The normalization kernel divides each exponential value by the computed sum.

4. **Shared Memory**:
   - Shared memory is used to store intermediate results during the reduction process.
   - This reduces the number of global memory accesses and improves performance.

5. **Error Checking**:
   - Added a `CUDA_CHECK` macro to validate CUDA API calls and catch errors early.
   - Implemented a `testSoftmax` function to verify that the softmax probabilities sum to 1.0 and are within the valid range [0, 1].
  

## Day 9
**File:** `attention.cu`

### **Summary**
Implemented the **attention mechanism** using CUDA. The attention mechanism is a key component in modern neural networks, particularly in transformers. It involves computing attention scores between queries and keys, applying softmax, and then using these scores to weight the values.

### **Key Concepts**
1. **Attention Mechanism**:
   - The attention mechanism computes a weighted sum of values, where the weights are determined by the compatibility of queries and keys.
   - The formula for attention is:
     
     $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$
     
   - Here, $Q$, $K$, and $V$ are the query, key, and value matrices, and $d$ is the dimensionality of the keys.

2. **CUDA Kernel for Attention Scores**:
   - Each thread computes the dot product between a query and a key, scaled by $\sqrt{d}$.
   - The results are stored in an attention scores matrix.

3. **Softmax Application**:
   - The softmax function is applied to the attention scores to convert them into probabilities.
   - This ensures that the attention weights sum to 1.

4. **Weighted Sum of Values**:
   - Each thread computes the weighted sum of values using the attention scores.
   - The result is stored in the output matrix.

5. **Verification**:
   - Verified the correctness of the implementation by manually computing the expected output for a specific element and comparing it with the GPU result.

## Day 10
**File:** `flash_attention.cu`

### **Summary**
Implemented the **Flash Attention mechanism** using CUDA. Flash Attention is an optimized version of the standard attention mechanism that reduces memory usage and improves computational efficiency by tiling the computation and avoiding redundant memory accesses. This implementation is still a work in progress and may produce incorrect outputs.

### **Key Concepts**
1. **Flash Attention**:
   - Flash Attention computes attention scores in a memory-efficient way by tiling the input matrices and processing them in chunks.
   - It avoids storing the full attention matrix, reducing memory overhead.

2. **Tiling**:
   - The input matrices `Q`, `K`, and `V` are divided into smaller tiles that fit into shared memory.
   - Each thread block processes one tile at a time, computing partial results.

3. **Online Softmax**:
   - Softmax is computed incrementally across tiles to avoid storing the full attention matrix.
   - This requires tracking the maximum value and sum of exponentials across tiles.

4. **Weighted Sum**:
   - The weighted sum of values is computed incrementally using the attention scores.

## Day 11
**File:** `tanh.cu`

### Summary
Implemented the hyperbolic tangent (tanh) function using CUDA. Wrote a CUDA kernel to perform an element-wise tanh operation on an array, where each thread computes the tanh of a single element.

### Learned
- **Element-wise Operations:** Understood how to implement element-wise mathematical functions using CUDA.
- **Kernel Launch Configuration:** Configured the number of threads per block and the number of blocks per grid for efficient parallel execution.
- **Device Memory Management:**
   - Allocated device memory using `cudaMalloc`.
   - Copied data between host and device using `cudaMemcpy`.
   - Freed device memory using `cudaFree`.
- **Verification:** Compared the results computed by the device with those computed by the host to ensure correctness.

## Day 12
**File:** `flash_attention_v2.cu`

### Summary
Implemented an initial version of the Flash Attention mechanism using CUDA, building on the concepts from Day 10. This version does not yet include tiling, which will be implemented later. The goal is to optimize memory usage and computational efficiency by avoiding redundant memory accesses.

### Key Concepts
1. **Flash Attention**:
    - Flash Attention computes attention scores in a memory-efficient way by processing input matrices in chunks.
    - This version focuses on the basic implementation without tiling.

2. **Attention Scores**:
    - Each thread computes the dot product between a query and a key, scaled by $\sqrt{d}$.
    - The results are stored in an attention scores matrix.

3. **Softmax Application**:
    - The softmax function is applied to the attention scores to convert them into probabilities.
    - This ensures that the attention weights sum to 1.

4. **Weighted Sum of Values**:
    - Each thread computes the weighted sum of values using the attention scores.
    - The result is stored in the output matrix.

5. **Future Work**:
    - Implement tiling to further optimize memory usage and computational efficiency.
    - Verify the correctness of the implementation by comparing the results with those from the standard attention mechanism.

## Day 13
**File:** `GELU.cu`

### Summary
Implemented the GELU (Gaussian Error Linear Unit) activation function using CUDA. The GELU function is used in neural networks to introduce non-linearity. This implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **GELU Activation Function**:
   - The GELU function is defined as:
     
     $\text{GELU}(x) = x \cdot 0.5 \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$
     
   - It smoothly approximates the ReLU function and is used in various deep learning models.

2. **CUDA Kernel for GELU**:
   - Each thread computes the GELU activation for a single element of the input array.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the GELU function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

5. **Future Work**:
   - Optimize the CUDA kernel for better performance.
   - Explore other activation functions and their CUDA implementations.
   
## Day 14
   **File:** `batchnorm.cu`

   ### Summary
   Implemented batch normalization using CUDA. Batch normalization is a technique to improve the training of deep neural networks by normalizing the inputs of each layer. This implementation includes both a CUDA kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Batch Normalization**:
      - Batch normalization normalizes the input to a layer by subtracting the mean and dividing by the standard deviation, followed by scaling and shifting.
      - The formula for batch normalization is:
        
        $\text{BN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
        
      - Here, $ \mu $ is the mean, $ \sigma^2 $ is the variance, $ \epsilon $ is a small constant to avoid division by zero, $ \gamma $ is the scale parameter, and $ \beta $ is the shift parameter.

   2. **CUDA Kernel for Batch Normalization**:
      - Each thread computes the normalized value for a single element of the input array.
      - The results are scaled and shifted using the $ \gamma $ and $ \beta $ parameters and stored in the output array.

   3. **CPU Implementation**:
      - A CPU version of the batch normalization function is implemented for comparison with the GPU results.
      - This helps in verifying the correctness of the CUDA implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the CPU and GPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

   5. **Future Work**:
      - Optimize the CUDA kernel for better performance.
      - Explore other normalization techniques and their CUDA implementations.

## Day 15
**File:** `flash_attention_v3.cu`

### Summary
Implemented an advanced version of the Flash Attention mechanism using CUDA. This version includes tiling and shared memory optimizations to further reduce memory usage and improve computational efficiency.

### Key Concepts
1. **Flash Attention**:
    - Flash Attention computes attention scores in a memory-efficient way by tiling the input matrices and processing them in chunks.
    - It avoids storing the full attention matrix, reducing memory overhead.

2. **Tiling and Shared Memory**:
    - The input matrices `Q`, `K`, and `V` are divided into smaller tiles that fit into shared memory.
    - Each thread block processes one tile at a time, computing partial results.
    - Shared memory is used to store intermediate results, reducing global memory accesses.

3. **Online Softmax**:
    - Softmax is computed incrementally across tiles to avoid storing the full attention matrix.
    - This requires tracking the maximum value and sum of exponentials across tiles.

4. **Weighted Sum**:
    - The weighted sum of values is computed incrementally using the attention scores.
    - Each thread computes the weighted sum of values using the attention scores and stores the result in the output matrix.

5. **Performance Comparison**:
    - Measured the execution time of the advanced Flash Attention implementation.
    - Compared the results with previous versions to evaluate the performance improvements.
    - Verified the correctness of the implementation by comparing the results with those from the standard attention mechanism.

6. **Future Work**:
    - Further optimize the CUDA kernel for better performance.
    - Explore other attention mechanisms and their CUDA implementations.

## Day 16
**File:** `conv2dtranspose.cu`

### Summary
Implemented the transposed convolution (also known as deconvolution) operation using CUDA. This operation is commonly used in neural networks for tasks such as image generation and upsampling.

### Key Concepts
1. **Transposed Convolution**:
   - The transposed convolution operation is used to upsample an input feature map.
   - It involves reversing the forward and backward passes of a standard convolution.

2. **CUDA Kernel for Transposed Convolution**:
   - Each thread computes the value of a single element in the output feature map.
   - The kernel iterates over the input feature map and the convolution kernel to compute the output values.

3. **Handling Stride and Padding**:
   - Implemented stride and padding to control the spatial dimensions of the output feature map.
   - Ensured that the kernel handles edge cases correctly to avoid accessing out-of-bounds memory.

4. **CPU Implementation**:
   - A CPU version of the transposed convolution function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

5. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

6. **Verification and Debugging**:
   - Verified the correctness of the output by comparing the results with those computed by the CPU.
   - Printed a small portion of the output feature map for visual inspection.

7. **Future Work**:
   - Optimize the CUDA kernel for better performance.
   - Explore other convolution operations and their CUDA implementations.

## Day 17
**File:** `geglu.cu`

### Summary
Implemented the GEGLU (Gated Linear Unit with GELU activation) operation using CUDA. This operation is used in neural networks to introduce non-linearity and gating mechanisms.

### Key Concepts
1. **GEGLU Activation Function**:
   - The GEGLU function combines the GELU activation with a gating mechanism.
   - The formula for GEGLU is:
     
     $\text{GEGLU}(x, y) = \text{GELU}(x) \cdot y$
     
   - Here, $x$ and $y$ are the input tensors, and GELU is the Gaussian Error Linear Unit activation function.

2. **CUDA Kernel for GEGLU**:
   - Each thread computes the GEGLU activation for a pair of elements from the input arrays.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the GEGLU function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

5. **Verification and Debugging**:
   - Verified the correctness of the output by comparing the results with those computed by the CPU.
   - Printed a small portion of the output array for visual inspection.

6. **Future Work**:
   - Optimize the CUDA kernel for better performance.
   - Explore other gating mechanisms and their CUDA implementations.

## Day 18
**File:** `swiglu.cu`

### Summary
Implemented the SwiGLU (Swish Gated Linear Unit) operation using CUDA. This operation is used in neural networks to introduce non-linearity and gating mechanisms, combining the Swish activation function with a gating mechanism. The file is a work on progress and can output incorrect results.

### Key Concepts
1. **SwiGLU Activation Function**:
   - The SwiGLU function combines the Swish activation with a gating mechanism.
   - The formula for SwiGLU is:
     
     $\text{SwiGLU}(x, y) = \text{Swish}(x) \cdot y$
     
   - Here, $x$ and $y$ are the input tensors, and Swish is the activation function defined as $x \cdot \text{sigmoid}(\beta x)$.

2. **CUDA Kernel for SwiGLU**:
   - Each thread computes the SwiGLU activation for a pair of elements from the input arrays.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the SwiGLU function is implemented for comparison with the GPU results.

## Day 19
**File:** `inverse_matrix.cu`

### Summary
Implemented matrix inversion using CUDA. This operation is essential in various numerical and scientific computations. The implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **Matrix Inversion**:
   - Matrix inversion involves finding a matrix that, when multiplied with the original matrix, yields the identity matrix.
   - The process includes augmenting the matrix with the identity matrix, performing row operations to convert the original matrix to the identity matrix, and applying the same operations to the augmented part to obtain the inverse.

2. **CUDA Kernels for Matrix Inversion**:
   - **Initialization Kernel**: Initializes the augmented matrix with the original matrix and the identity matrix.
   - **Row Swap Kernel**: Swaps rows to ensure numerical stability.
   - **Normalization Kernel**: Normalizes the pivot row.
   - **Elimination Kernel**: Eliminates the current column in other rows.

3. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results
- **Matrix Size: 256x256**
  - Max Error: 4.27751e-06
  - CPU Time: 0.146624 s
  - GPU Time: 0.0140411 s
  - Speedup: 10.4425x

- **Matrix Size: 512x512**
  - Max Error: 1.02124e-06
  - CPU Time: 1.18075 s
  - GPU Time: 0.0498691 s
  - Speedup: 23.6771x

- **Matrix Size: 1024x1024**
  - Max Error: 2.51715e-07
  - CPU Time: 10.1626 s
  - GPU Time: 0.268092 s
  - Speedup: 37.9073x

### Future Work
- Optimize the CUDA kernels for better performance.
- Explore other matrix operations and their CUDA implementations.
- Investigate numerical stability and precision improvements.
- Validate the implementation with larger matrices and different types of matrices.

## Day 20
**File:** `kmeans.cu`

### Summary
Implemented the K-Means clustering algorithm using CUDA. This algorithm partitions a set of data points into clusters, where each point belongs to the cluster with the nearest mean. The implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **K-Means Clustering**:
   - The K-Means algorithm partitions data points into $K$ clusters by iteratively updating cluster centroids and reassigning points to the nearest centroid.
   - The goal is to minimize the sum of squared distances between points and their assigned cluster centroids.

2. **CUDA Kernels for K-Means**:
   - **Assign Clusters Kernel**: Each thread computes the distance between a data point and all centroids, assigning the point to the nearest centroid.
   - **Update Centroids Kernel**: Each thread updates the centroids by computing the mean of all points assigned to each cluster.

3. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results
- **K-Means Clustering Results**
  - Data Points: 1000000
  - Dimensions: 2
  - Clusters: 3
  - Validation: PASSED
  - CPU Time: 3.26551s
  - GPU Time: 0.310679s
  - Speedup: 10.5109x

### Future Work
- Optimize the CUDA kernels for better performance.
- Explore other clustering algorithms and their CUDA implementations.
- Investigate the impact of different initialization methods on clustering performance.
- Validate the implementation with higher-dimensional data and more clusters.

## Day 21
**File:** `diffusion_noising.cu`

### Summary
Implemented the diffusion noising process using CUDA. This process is used in various machine learning models, particularly in generative models, to add noise to data in a controlled manner. The implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **Diffusion Noising**:
   - The diffusion noising process involves adding noise to data in a way that is controlled by a parameter, typically used in generative models.
   - The formula for diffusion noising is:
     
     $x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1 - \alpha_t} \cdot \text{noise}$
     
   - Here, $x_t$ is the noisy data at timestep $t$, $\alpha_t$ is a parameter controlling the amount of noise, and `noise` is a random Gaussian noise.

2. **CUDA Kernel for Diffusion Noising**:
   - Each thread computes the noisy value for a single element of the input array.
   - The kernel uses the provided noise and alpha parameters to compute the noisy data.

3. **CPU Implementation**:
   - A CPU version of the diffusion noising function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results
- **Diffusion Noising Results**
  - Validation: PASSED
  - CPU Time: 0.003534 seconds
  - GPU Time: 0.000190464 seconds
  - Speedup: 18.5547x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore other noising techniques and their CUDA implementations.
- Investigate the impact of different noise distributions on the diffusion process.
- Validate the implementation with different datasets and parameters.