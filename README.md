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

## Day 22
**File:** `elish.cu`

### Summary
Implemented the ELiSH (Exponential Linear Squashing) activation function using CUDA. The ELiSH function is used in neural networks to introduce non-linearity. This implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **ELiSH Activation Function**:
   - The ELiSH function is defined as:
     
     $\text{ELiSH}(x) = \begin{cases} 
     x \cdot \text{sigmoid}(x) & \text{if } x \geq 0 \\
     (\exp(x) - 1) \cdot \text{sigmoid}(x) & \text{if } x < 0 
     \end{cases}$
     
   - It combines the exponential and sigmoid functions to create a smooth, non-linear activation.

2. **CUDA Kernel for ELiSH**:
   - Each thread computes the ELiSH activation for a single element of the input array.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the ELiSH function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results
- **ELiSH Activation Results**
  - Validation: PASSED
  - CPU Time: 0.0186454 seconds
  - GPU Time: 0.000240544 seconds
  - Speedup: 77.5135x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore other activation functions and their CUDA implementations.
- Investigate the impact of ELiSH on different neural network architectures.
- Validate the implementation with larger datasets and different parameters.

## Day 23
**File:** `selu.cu`

### Summary
Implemented the SELU (Scaled Exponential Linear Unit) activation function using CUDA. The SELU function is used in neural networks to introduce non-linearity and maintain self-normalizing properties. This implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **SELU Activation Function**:
   - The SELU function is defined as:
     
     $\text{SELU}(x) = \lambda \begin{cases} 
     x & \text{if } x \geq 0 \\ 
     \alpha (\exp(x) - 1) & \text{if } x < 0 
     \end{cases}$
     
   - Here, $\lambda$ and $\alpha$ are predefined constants that ensure the self-normalizing property of the activation function.

2. **CUDA Kernel for SELU**:
   - Each thread computes the SELU activation for a single element of the input array.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the SELU function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results
- **SELU Activation Results**
  - Validation: PASSED
  - CPU Time: 0.0139118 seconds
  - GPU Time: 0.00019568 seconds
  - Speedup: 71.0947x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore other activation functions and their CUDA implementations.
- Investigate the impact of SELU on different neural network architectures.
- Validate the implementation with larger datasets and different parameters.

## Day 24
**File:** `adam.cu`

### Summary
Implemented the Adam optimization algorithm using CUDA. Adam is an adaptive learning rate optimization algorithm widely used in training deep learning models. This implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **Adam Optimization Algorithm**:
   - Adam combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.
   - The update rule for Adam is:
     
     $m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$
     
     $v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$
     
     $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
     
     $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
     
     $\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
     
   - Here, $m_t$ and $v_t$ are the first and second moment estimates, $\beta_1$ and $\beta_2$ are the decay rates, $\eta$ is the learning rate, and $\epsilon$ is a small constant to prevent division by zero.

2. **CUDA Kernel for Adam**:
   - Each thread updates the parameters for a single element of the input array using the Adam update rule.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the Adam optimization algorithm is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results
- **Validation Results**
  - Parameters: PASSED
  - First moment (m): PASSED
  - Second moment (v): PASSED

- **Performance**
  - CPU time: 0.0780394 seconds
  - GPU time: 0.000322752 seconds
  - Speedup: 241.794x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore other optimization algorithms and their CUDA implementations.
- Investigate the impact of Adam on different neural network architectures.
- Validate the implementation with larger datasets and different parameters.

## Day 25
**File:** `quantization.cu`

### Summary
Implemented quantization and dequantization using CUDA. Quantization is a technique used to reduce the precision of model parameters and activations, which is crucial for efficient deployment of deep learning models, especially on resource-constrained devices. This implementation includes both CUDA kernels and CPU versions for comparison.

### Key Concepts
1. **Quantization**:
   - Quantization maps floating-point values to a smaller set of discrete values, typically 8-bit integers (uint8).
   - The quantization formula is:
     
     $$Q(x) = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$
     
   - Here, `scale` determines the quantization step size, and `zero_point` shifts the quantized values to center them around zero.

2. **Dequantization**:
   - Dequantization maps the quantized values back to floating-point values.
   - The dequantization formula is:
     
     $$D(q) = (q - \text{zero\_point}) \times \text{scale}$$
     
3. **CUDA Kernels for Quantization and Dequantization**:
   - Each thread processes a single element of the input array.
   - The `quantize_kernel` converts FP32 values to uint8 using the quantization formula.
   - The `dequantize_kernel` converts uint8 values back to FP32 using the dequantization formula.

4. **CPU Implementation**:
   - A CPU version of the quantization and dequantization process is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

5. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results
- **Validation Results**
  - Quantization: PASSED
  - Dequantization error: WITHIN TOLERANCE

- **Performance**
   - Performance (quantization + dequantization):
   - CPU time: 0.27055 seconds
   - GPU time: 0.00140893 seconds
   - Speedup: 192.025x

### Future Work
- Optimize the CUDA kernels for better performance.
- Explore different quantization schemes, such as symmetric vs. asymmetric quantization.
- Investigate the impact of quantization on model accuracy and performance.
- Validate the implementation with larger datasets and different quantization parameters.
- Implement mixed-precision quantization for more efficient model deployment.

## Day 26
**File:** `quantization_v2.cu`

### Summary
Implemented an optimized version of quantization using CUDA with shared memory and tiling. Quantization is a technique used to reduce the precision of model parameters and activations, which is crucial for efficient deployment of deep learning models, especially on resource-constrained devices. This implementation includes both CUDA kernels and CPU versions for comparison.

### Key Concepts
1. **Quantization**:
   - Quantization maps floating-point values to a smaller set of discrete values, typically 8-bit integers (uint8).
   - The quantization formula is:
     
     $$Q(x) = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$

2. **CUDA Kernel with Shared Memory and Tiling**:
   - Utilized shared memory to load data into tiles, reducing global memory access latency.
   - Each thread processes a single element of the input array within its tile.
   - The `quantize_kernel` converts FP32 values to uint8 using the quantization formula.

3. **CPU Implementation**:
   - A CPU version of the quantization process is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
  - CPU time: 0.210664 seconds
  - GPU time: 0.000843776 seconds
  - Speedup: 249.668x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore different quantization schemes, such as symmetric vs. asymmetric quantization.
- Investigate the impact of quantization on model accuracy and performance.
- Validate the implementation with larger datasets and different quantization parameters.
- Implement mixed-precision quantization for more efficient model deployment.


## Day 27
**File:** `swish.cu`

### Summary
Implemented the Swish activation function using CUDA. The Swish function is used in neural networks to introduce non-linearity and has been shown to perform better than ReLU in some cases. This implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **Swish Activation Function**:
   - The Swish function is defined as:
     
     $\text{Swish}(x) = x \cdot \text{sigmoid}(x)$
     
   - It combines the input with the sigmoid function to create a smooth, non-linear activation.

2. **CUDA Kernel for Swish**:
   - Each thread computes the Swish activation for a single element of the input array.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the Swish function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
  - CPU time: 0.234491 seconds
  - GPU time: 0.000752832 seconds
  - Speedup: 311.479x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore other activation functions and their CUDA implementations.

## Day 28
**File:** `elu.cu`

### Summary
Implemented the Exponential Linear Unit (ELU) activation function using CUDA. The ELU function is used in neural networks to introduce non-linearity and can help mitigate the vanishing gradient problem. This implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **ELU Activation Function**:
   - The ELU function is defined as:
     
     $$\text{ELU}(x) = \begin{cases} 
     x & \text{if } x > 0 \\
     \alpha (\exp(x) - 1) & \text{if } x \leq 0 
     \end{cases}$$
     
   - It introduces a smooth, non-linear activation that can help improve learning in deep neural networks.

2. **CUDA Kernel for ELU**:
   - Each thread computes the ELU activation for a single element of the input array.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the ELU function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
  - CPU time: 0.234491 seconds
  - GPU time: 0.000752832 seconds
  - Speedup: 311.479x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore other activation functions and their CUDA implementations.


## Day 29
**File:** `smooth_swiglu.cu`

### Summary
Implemented the Smooth SwiGLU activation function using CUDA. The Smooth SwiGLU function is used in neural networks to introduce non-linearity and gating mechanisms, combining the Swish activation function with a gating mechanism. This implementation includes both a CUDA kernel and a CPU version for comparison.

### Key Concepts
1. **Smooth SwiGLU Activation Function**:
   - The Smooth SwiGLU function combines the Swish activation with a gating mechanism.
   - The formula for Smooth SwiGLU is:
     
     $$\text{Smooth SwiGLU}(x) = \frac{\text{SwiGLU}(x)}{1 + \exp(-\text{SwiGLU}(x))}$$
     
   - Here, SwiGLU is defined in day 18.

2. **CUDA Kernel for Smooth SwiGLU**:
   - Each thread computes the Smooth SwiGLU activation for a single element of the input array.
   - The results are stored in the output array.

3. **CPU Implementation**:
   - A CPU version of the Smooth SwiGLU function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUDA implementation.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
  - CPU time: 0.41193 seconds
  - GPU time: 0.000925408 seconds
  - Speedup: 445.133x

### Future Work
- Optimize the CUDA kernel for better performance.
- Explore other activation functions and their CUDA implementations.
- Validate the implementation with larger datasets and different parameters.


## Day 30
**File:** `cutlass_prod.cu`

### Summary
Explored the CUTLASS library to implement matrix multiplication using CUDA. The CUTLASS library provides highly optimized CUDA kernels for various linear algebra operations. This implementation includes both a CUDA kernel using CUTLASS and a CPU version for comparison.

### Key Concepts
1. **CUTLASS Library**:
   - CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is a collection of CUDA C++ templates for implementing high-performance matrix-multiplication (GEMM) and related computations.
   - It provides a flexible and efficient way to perform matrix operations on NVIDIA GPUs.

2. **Matrix Multiplication**:
   - Matrix multiplication involves computing the product of two matrices, resulting in a third matrix.
   - The formula for matrix multiplication is:
     
     $C = A \times B$
     
   - Here, $A$, $B$, and $C$ are matrices, and the product is computed by taking the dot product of rows of $A$ with columns of $B$.

3. **CUDA Kernel using CUTLASS**:
   - Utilized the CUTLASS library to perform matrix multiplication on the GPU.
   - Configured the CUTLASS GEMM operation with appropriate parameters for matrix dimensions and data types.

4. **CPU Implementation**:
   - A CPU version of the matrix multiplication function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUTLASS implementation.

5. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Validation**: PASSED
- **Performance**
  - CPU time: 8.00976 seconds
  - GPU time: 0.00133392 seconds
  - Speedup: 6004.68x

### Future Work
- Explore other operations provided by the CUTLASS library.
- Optimize the CUTLASS configuration for better performance.
- Validate the implementation with larger matrices and different configurations.


## Day 31
**File:** `batched_prod.cu`

### Summary
Implemented batched matrix multiplication using CUDA and the CUTLASS library. This implementation includes both a CUDA kernel using CUTLASS for batched GEMM (General Matrix Multiplication) and a CPU version for comparison.

### Key Concepts
1. **CUTLASS Library**:
   - CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is a collection of CUDA C++ templates for implementing high-performance matrix-multiplication (GEMM) and related computations.
   - It provides a flexible and efficient way to perform batched GEMM operations on NVIDIA GPUs.

2. **Batched Matrix Multiplication**:
   - Batched matrix multiplication involves performing multiple matrix multiplications in parallel.
   - The formula for matrix multiplication is:
     
     $C = \alpha \cdot A \cdot B + \beta \cdot C$
     
   - Here, $A$, $B$, and $C$ are matrices, and $\alpha$ and $\beta$ are scalars.

3. **CUDA Kernel using CUTLASS**:
   - Utilized the CUTLASS library to perform batched GEMM on the GPU.
   - Configured the CUTLASS GEMM operation with appropriate parameters for matrix dimensions, data types, and batch size.

4. **CPU Implementation**:
   - A CPU version of the batched GEMM function is implemented for comparison with the GPU results.
   - This helps in verifying the correctness of the CUTLASS implementation.

5. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
  - CPU time: 67.243 seconds
  - GPU time: 0.004 seconds
  - Speedup: 16810.75x
  - Maximum error: 0.00012207

### Future Work
- Explore other batched operations provided by the CUTLASS library.
- Optimize the CUTLASS configuration for better performance.
- Validate the implementation with larger matrices and different configurations.
- Investigate the impact of different matrix sizes and batch counts on performance.
- Explore the use of CUTLASS for other deep learning operations.

## Day 32
**File:** `spmv.cu`

### Summary
Implemented **Sparse Matrix-Vector Multiplication (SpMV)** using CUDA. This implementation includes a CUDA kernel for SpMV using the **Compressed Sparse Row (CSR)** format and a CPU version for validation and performance comparison.

### Key Concepts
1. **Sparse Matrix Representation**:
   - Sparse matrices are represented using the **Compressed Sparse Row (CSR)** format, which consists of three arrays:
     - `values`: Stores the non-zero values of the matrix.
     - `col_indices`: Stores the column indices of the non-zero values.
     - `row_ptr`: Stores the starting index of each row in the `values` and `col_indices` arrays.

2. **Sparse Matrix-Vector Multiplication (SpMV)**:
   - SpMV computes the product of a sparse matrix $A$ and a dense vector $x$ to produce a dense vector $y$:
     
     $y = A \cdot x$
     
   - In CSR format, this involves iterating over the non-zero elements of each row and computing the dot product with the corresponding elements of $x$.

3. **CUDA Kernel for SpMV**:
   - Each thread in the CUDA kernel computes one element of the output vector $y$ by processing the non-zero elements of the corresponding row in the sparse matrix.
   - The kernel uses global memory to access the sparse matrix and input vector.

4. **CPU Implementation**:
   - A CPU version of SpMV is implemented for validation and performance comparison.
   - This ensures the correctness of the CUDA kernel and provides a baseline for performance evaluation.

5. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
    -  CPU time: 0.00684534 seconds
    -  GPU time: 0.000342016 seconds
    -  Speedup: 20.0147x


### Future Work
- **Optimize Kernel**:
  - Use shared memory to reduce global memory accesses.
  - Implement warp-level parallelism to improve performance.
- **Larger Problem Sizes**:
  - Test the implementation with larger sparse matrices to evaluate scalability.
- **Use cuSPARSE**:
  - Compare the custom CUDA kernel with NVIDIA's `cuSPARSE` library for SpMV.
- **Different Sparse Formats**:
  - Explore other sparse matrix formats like **Compressed Sparse Column (CSC)** or **ELLPACK**.
- **Application to Real-World Problems**:
  - Apply SpMV to real-world problems such as graph algorithms or iterative solvers for linear systems.


## Day 33
**File:** `cross_entropy_loss.cu`

### Summary
Implemented **Cross-Entropy Loss** using CUDA. This implementation includes a CUDA kernel for computing the loss and a CPU version for validation and performance comparison.

### Key Concepts
1. **Cross-Entropy Loss**:
   - A loss function commonly used in classification tasks.
   - Measures the difference between predicted probabilities and ground truth labels.

2. **CUDA Kernel**:
   - Each thread computes the loss for one sample in the batch.
   - Uses global memory to access predictions and labels.

3. **CPU Implementation**:
   - A CPU version of the loss function is implemented for validation and performance comparison.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
  - CPU time: 0.00157762 seconds
  - GPU time: 0.000155296 seconds
  - Speedup: 10.1588x

---

### Analysis of Speedup

The GPU implementation achieves a **speedup of ~10x** over the CPU implementation. While this is a significant improvement, the speedup is relatively modest compared to what GPUs are capable of (often 100x or more for highly parallelizable tasks). Here are the key reasons for the limited speedup:

---

#### 1. **Small Problem Size**:
   - The problem size (`num_samples = 100000`, `num_classes = 10`) is still relatively small for a GPU.
   - GPUs excel at handling massive parallelism, but for smaller problems, the overhead of launching the kernel and transferring data between the CPU and GPU can dominate the execution time.

---

#### 2. **Memory Bandwidth Bottleneck**:
   - The kernel accesses global memory in a non-coalesced manner, which can lead to poor memory bandwidth utilization.
   - Each thread reads the `predictions` and `labels` arrays from global memory, which is much slower than accessing shared memory or registers.
   - Optimizing memory access patterns (e.g., using shared memory or coalesced memory accesses) could improve performance.

---

#### 3. **Kernel Overhead**:
   - The kernel is relatively simple, with each thread performing only a few arithmetic operations (logarithm and addition).
   - The overhead of launching the kernel and managing threads can outweigh the actual computation, especially for small problem sizes.

---

#### 4. **Data Transfer Overhead**:
   - The time taken to transfer data between the CPU and GPU (`cudaMemcpy`) is included in the GPU timing.
   - For small to medium-sized problems, this overhead can significantly impact the overall performance.
   - Overlapping data transfer with computation using CUDA streams or asynchronous memory copies could mitigate this issue.

---

#### 5. **Limited Parallelism**:
   - The kernel assigns one thread per sample, which may not fully utilize the GPU's parallel processing capabilities.


---

### Future Work
1. **Optimize Kernel**:
   - Use shared memory to reduce global memory accesses.
   - Implement warp-level parallelism to improve performance.
   - Process multiple samples per thread to increase parallelism.

2. **Increase Problem Size**:
   - Test the implementation with larger batch sizes (e.g., `num_samples = 1,000,000`) to better utilize the GPU's parallel processing power.

3. **Reduce Data Transfer Overhead**:
   - Overlap data transfer with computation using CUDA streams or asynchronous memory copies.



## Day 34
**File:** `perplexity.cu`

### Summary
Implemented **Perplexity**, a metric used in language models, using CUDA. This implementation includes a CUDA kernel for computing perplexity and a CPU version for validation and performance comparison.

### Key Concepts
1. **Perplexity**:
   - A metric used to evaluate language models.
   - Defined as:
     
     $$
     \text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log(p_i)\right)
     $$
     
   - Lower perplexity indicates better model performance.

2. **CUDA Kernel**:
   - Each thread computes the log loss for one sample.
   - Uses global memory to access probabilities and labels.

3. **CPU Implementation**:
   - A CPU version of the perplexity computation is implemented for validation and performance comparison.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
   - CPU time: 0.00157565 seconds
   - GPU time: 0.000169376 seconds
   - Speedup: 9.3027x

### Future Work
- **Optimize Kernel**:
  - Use shared memory to reduce global memory accesses.
  - Implement warp-level parallelism to improve performance.
- **Larger Problem Sizes**:
  - Test the implementation with larger datasets to evaluate scalability.
- **Application to Real-World Problems**:
  - Integrate perplexity computation into a language model training pipeline.


## Day 35
**File:** `attention_cutlass.cu`

### Summary
Implemented **Self-Attention Mechanism** using CUDA and CUTLASS. This implementation includes CUDA kernels for softmax and weighted aggregation, and uses CUTLASS for matrix multiplication. This a work on progress which may produce some errors.

### Key Concepts
1. **Self-Attention**:
   - A mechanism used in transformer models to compute attention scores, apply softmax, and perform weighted aggregation.
   - Defined as:
     
     $$
     \text{Scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}
     $$
     
     $$
     \text{Attention Weights} = \text{Softmax}(\text{Scores})
     $$
     
     $$
     \text{Output} = \text{Attention Weights} \cdot V
     $$

2. **CUTLASS**:
   - Used for efficient matrix multiplication (GEMM) on the GPU.

3. **CUDA Kernels**:
   - `softmax_kernel`: Applies softmax to the attention scores.
   - `weighted_aggregation_kernel`: Computes the weighted aggregation of values.

4. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.


### Future Work
- **Optimize Kernels**:
  - Use shared memory to reduce global memory accesses.
  - Implement warp-level parallelism to improve performance.
- **Larger Problem Sizes**:
  - Test the implementation with larger matrices to evaluate scalability.

(Due to technical issues, the search service is temporarily unavailable.)

## Day 36
**File:** `kl_divergence.cu`

### Summary
Implemented **Kullback-Leibler (KL) Divergence** using CUDA. This implementation includes a CUDA kernel for computing KL divergence and a CPU version for validation and performance comparison. The GPU implementation achieves a **104.5x speedup** over the CPU version.

### Key Concepts
1. **KL Divergence**:
   - A measure of how one probability distribution \( P \) diverges from a second, reference probability distribution \( Q \).
   - Defined as:
     
     $$
     D_{KL}(P || Q) = \sum_{i} P(i) \cdot \log\left(\frac{P(i)}{Q(i)}\right)
     $$
     
   - Lower KL divergence indicates that the distributions are more similar.

2. **CUDA Kernel**:
   - Each thread computes the contribution of one element to the KL divergence.
   - Uses global memory to access the distributions \( P \) and \( Q \).

3. **Performance Comparison**:
   - Measured the execution time of both the CPU and GPU implementations.
   - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

### Results

- **Performance**
  - CPU time: 0.0132517 seconds
  - GPU time: 0.000126816 seconds
  - Speedup: 104.5x
  - Validation: PASSED (error < 1e-5)

---

### Use Cases of KL Divergence

#### 1. **Diffusion Models**
   - KL divergence is widely used in diffusion models, such as **Denoising Diffusion Probabilistic Models (DDPMs)**, to measure the difference between the predicted noise distribution and the true noise distribution at each step of the diffusion process.
   - Example: In DDPMs, the loss function often includes a KL divergence term to ensure that the predicted noise matches the true noise distribution.

   **Reference**:
   - Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *arXiv preprint arXiv:2006.11239*.

#### 2. **Variational Autoencoders (VAEs)**
   - KL divergence is a key component of the loss function in VAEs, where it measures the difference between the learned latent distribution and a prior distribution (e.g., a standard Gaussian).
   - Example: In VAEs, minimizing the KL divergence ensures that the latent space is well-structured and follows the desired prior distribution.

   **Reference**:
   - Kingma, D. P., & Welling, M. (2013). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). *arXiv preprint arXiv:1312.6114*.

#### 3. **Reinforcement Learning**
   - KL divergence is used in reinforcement learning algorithms, such as **Trust Region Policy Optimization (TRPO)** and **Proximal Policy Optimization (PPO)**, to constrain policy updates and ensure stable training.
   - Example: In PPO, KL divergence is used to limit the difference between the old and new policies during optimization.

   **Reference**:
   - Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477). *arXiv preprint arXiv:1502.05477*.


---

### Future Work
1. **Optimize Kernel**:
   - Use shared memory to reduce global memory accesses.
   - Implement warp-level parallelism to improve performance.
2. **Larger Problem Sizes**:
   - Test the implementation with larger datasets to evaluate scalability.
3. **Application to Real-World Problems**:
   - Integrate KL divergence computation into machine learning pipelines, such as diffusion models or VAEs.

   ## Day 37
   **File:** `backward_mlp.cu`

   ### Summary
   Implemented the backward pass for a multi-layer perceptron (MLP) using CUDA. This implementation includes CUDA kernels for computing gradients of the ReLU activation function, output layer, hidden layer, and weights and biases. A CPU version is also provided for validation and performance comparison.

   ### Key Concepts
   1. **ReLU Derivative**:
      - Implemented a CUDA kernel to compute the derivative of the ReLU activation function.
      - Each thread computes the derivative for a single element of the input array.

   2. **Output Layer Gradients**:
      - Implemented a CUDA kernel to compute the gradients of the output layer.
      - Each thread computes the gradient for a single element of the output array.

   3. **Hidden Layer Gradients**:
      - Implemented a CUDA kernel to compute the gradients of the hidden layer.
      - Each thread computes the gradient for a single element of the hidden layer.

   4. **Gradients of Weights and Biases**:
      - Implemented a CUDA kernel to compute the gradients of the weights and biases.
      - Each thread computes the gradient for a single weight or bias.

   5. **CPU Implementation**:
      - A CPU version of the backward pass is implemented for comparison with the GPU results.
      - This helps in verifying the correctness of the CUDA implementation.

   6. **Performance Comparison**:
      - Measured the execution time of both the CPU and GPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

   ### Results

   - **Performance**
      - CPU time: 0.0233857 seconds
      - GPU time: 0.00613648 seconds
      - Speedup: 3.81092x

   ### Future Work
   - Optimize the CUDA kernels for better performance.
   - Explore other activation functions and their backward pass implementations.
   - Validate the implementation with larger datasets and different network architectures.

   ## Day 38
   **File:** `triton_add.py`

   ### Summary
   Implemented vector addition using Triton, a language and compiler for writing custom deep learning primitives. Compared the performance of the Triton kernel with PyTorch's built-in vector addition.

   ### Key Concepts
   1. **Triton Kernel**:
      - Wrote a Triton kernel for vector addition.
      - Each program processes a block of elements, loading data from global memory, performing the addition, and storing the result back to global memory.

   2. **Performance Comparison**:
      - Measured the execution time of both the Triton kernel and PyTorch's vector addition.
      - Compared the results to ensure they match and evaluated computation time.

   ### Results
   - **Performance**
     - Triton kernel time: 144.8014 ms
     - PyTorch vector addition time: 0.2902 ms
     - Validation passed!

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other operations and their Triton implementations.

   ## Day 39
   **File:** `triton_mat_vec_prod.py`

   ### Summary
   Implemented matrix-vector multiplication using Triton. Compared the performance of the Triton kernel with PyTorch's built-in matrix-vector multiplication.

   ### Key Concepts
   1. **Triton Kernel**:
      - Wrote a Triton kernel for matrix-vector multiplication.
      - Each program processes a block of rows, loading data from global memory, performing the dot product, and storing the result back to global memory.

   2. **Performance Comparison**:
      - Measured the execution time of both the Triton kernel and PyTorch's matrix-vector multiplication.
      - Compared the results to ensure they match and evaluated computation time.

   ### Results
   - **Performance**
     - Triton kernel time: 4783.3338 ms
     - PyTorch matrix-vector multiplication time: 151.9542 ms

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other operations and their Triton implementations.

   ## Day 40
   **File:** `matrix_product.py`

   ### Summary
   Implemented matrix multiplication using Triton. Compared the performance of the Triton kernel with PyTorch's built-in matrix multiplication.

   ### Key Concepts
   1. **Triton Kernel**:
      - Wrote a Triton kernel for matrix multiplication.
      - Each program processes a block of elements, loading data from global memory, performing the multiplication, and storing the result back to global memory.

   2. **Performance Comparison**:
      - Measured the execution time of both the Triton kernel and PyTorch's matrix multiplication.
      - Compared the results to ensure they match and evaluated computation time.

   ### Results
   - **Performance**
      - Triton kernel time: 3.8714 ms
      - PyTorch (GPU) matrix multiplication time: 0.3941 ms
      - PyTorch (CPU) matrix multiplication time: 228.8835 ms

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other operations and their Triton implementations.
   - Explore the use of Triton for other deep learning operations.

   ## Day 41
   **File:** `triton_softmax.py`

   ### Summary
   Implemented the softmax function using Triton. The softmax function is widely used in machine learning to convert a vector of raw scores (logits) into probabilities. Compared the performance of the Triton kernel with PyTorch's built-in softmax function.

   ### Key Concepts
   1. **Softmax Function**:
      - Converts logits into probabilities.
      - Formula:
        
        $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

   2. **Triton Kernel**:
      - Wrote Triton kernels for computing the maximum value per row and the softmax function.
      - Each program processes a block of elements, loading data from global memory, performing the necessary computations, and storing the result back to global memory.

   3. **Performance Comparison**:
      - Measured the execution time of both the Triton kernel and PyTorch's softmax function.
      - Compared the results to ensure they match and evaluated computation time.

   ### Results
   - **Performance**
     - Triton kernel time: 5.8584 ms
     - PyTorch (GPU) Softmax time: 0.9866 ms
     - Validation passed!

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other operations and their Triton implementations.

   ## Day 42
   **File:** `triton_relu.py`

   ### Summary
   Implemented the ReLU (Rectified Linear Unit) function using Triton. Compared the performance of the Triton kernel with a CPU implementation.

   ### Results
   - **Performance**
      - Triton Execution Time: 0.002692 seconds
      - CPU Execution Time: 0.039735 seconds
   - Validation passed!

   ## Day 43
   **File:** `triton_silu.py`

   ### Summary
   Implemented the SiLU (Sigmoid Linear Unit) function using Triton. Compared the performance of the Triton kernel with a CPU implementation.

 
    **Performance**
      - Triton Execution Time: 0.003542 seconds
      - CPU Execution Time: 0.612337 seconds

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other activation functions and their Triton implementations.
   - Validate the implementation with larger datasets and different parameters.

   ## Day 44
   **File:** `triton_gelu.py`

   ### Summary
   Implemented the GELU (Gaussian Error Linear Unit) activation function using Triton. The GELU function is used in neural networks to introduce non-linearity. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **GELU Activation Function**:
      - The GELU function is defined as:
        
        $\text{GELU}(x) = x \cdot 0.5 \cdot (1 + \text{erf}(x / \sqrt{2}))$
        
      - It smoothly approximates the ReLU function and is used in various deep learning models.

   2. **Triton Kernel for GELU**:
      - Each program processes a block of elements, loading data from global memory, performing the GELU computation, and storing the result back to global memory.

   3. **CPU Implementation**:
      - A CPU version of the GELU function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Performance**
     - Triton Execution Time: 0.002354 seconds
     - CPU Execution Time: 0.779846 seconds

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other activation functions and their Triton implementations.
   - Validate the implementation with larger datasets and different parameters.

   ## Day 45
   **File:** `triton_swish.py`

   ### Summary
   Implemented the Swish activation function using Triton. The Swish function is used in neural networks to introduce non-linearity and has been shown to perform better than ReLU in some cases. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Swish Activation Function**:
      - The Swish function is defined as:
        
        $\text{Swish}(x) = x \cdot \text{sigmoid}(x)$
        
      - It combines the input with the sigmoid function to create a smooth, non-linear activation.

   2. **Triton Kernel for Swish**:
      - Each program processes a block of elements, loading data from global memory, performing the Swish computation, and storing the result back to global memory.

   3. **CPU Implementation**:
      - A CPU version of the Swish function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Performance**
     - Triton Execution Time: 0.003623 seconds
     - CPU Execution Time: 0.772183 seconds

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other activation functions and their Triton implementations.
   - Validate the implementation with larger datasets and different parameters.

   ## Day 46
   **File:** `triton_swiglu.py`

   ### Summary
   Implemented the SwiGLU (Swish Gated Linear Unit) activation function using Triton. The SwiGLU function is used in neural networks to introduce non-linearity and gating mechanisms, combining the Swish activation function with a gating mechanism. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **SwiGLU Activation Function**:
      - The SwiGLU function combines the Swish activation with a gating mechanism.
      - The formula for SwiGLU is:
        
        $\text{SwiGLU}(x, y) = \text{Swish}(x) \cdot y$
        
      - Here, $x$ and $y$ are the input tensors, and Swish is the activation function defined as $x \cdot \text{sigmoid}(x)$.

   2. **Triton Kernel for SwiGLU**:
      - Each program processes a block of elements, loading data from global memory, performing the SwiGLU computation, and storing the result back to global memory.

   3. **CPU Implementation**:
      - A CPU version of the SwiGLU function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Performance**
     - Triton Execution Time: 0.004779 seconds
     - CPU Execution Time: 0.159479 seconds

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other activation functions and their Triton implementations.
   - Validate the implementation with larger datasets and different parameters.

   ## Day 47
   **File:** `relu2.cu`

   ### Summary
   Implemented the ReLU² (Squared ReLU) activation function using CUDA. ReLU² squares the positive inputs while zeroing out negative values. This implementation includes both a CUDA kernel and a CPU version for comparison.

   ### Key Concepts
   1. **ReLU² Activation Function**:
      - The ReLU² function is defined as:
        
        $\text{ReLU}^2(x) = \begin{cases} 
        x^2 & \text{if } x > 0 \\
        0 & \text{if } x \leq 0 
        \end{cases}$
        
      - It introduces non-linearity while maintaining a steeper gradient for positive inputs compared to standard ReLU.

   2. **CUDA Kernel for ReLU²**:
      - Each thread computes the ReLU² activation for a single element of the input array.
      - The kernel evaluates whether the input is positive, and if so, squares it; otherwise, it outputs zero.

   3. **CPU Implementation**:
      - A CPU version of the ReLU² function is implemented for comparison with the GPU results.
      - This helps in verifying the correctness of the CUDA implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the CPU and GPU implementations.
      - GPU implementation achieved a substantial speedup over the CPU version.

   ### Advantages of ReLU² Over Other Activations

   1. **Enhanced Feature Learning**:
      - The squared output for positive values can help neural networks learn more expressive features compared to standard ReLU.
      - The quadratic behavior provides stronger activation for more significant features while still having the sparsity benefits of ReLU.

   2. **Improved Gradient Flow**:
      - ReLU² has a larger gradient for positive inputs compared to standard ReLU, potentially leading to faster learning for strongly activated neurons.
      - The gradient increases linearly with the input value, unlike ReLU where the gradient is constant.

   3. **Maintaining Sparsity**:
      - Like standard ReLU, ReLU² induces sparsity in neural networks by outputting zero for negative inputs, which can help with regularization.

   4. **Computational Efficiency**:
      - ReLU² is computationally efficient, involving only a comparison and a multiplication operation.

   ### Results
   - **Performance**
     - CPU Execution Time: 9.94267 ms
     - GPU Execution Time: 0.160084 ms
     - Speedup: ~62x

   ### Future Work
   - Investigate the impact of ReLU² on model convergence and accuracy.
   - Explore techniques to address potential numerical stability issues with large positive inputs.

   ## Day 48
   **File:** `qlora_quant.cu`

   ### Summary
   Implemented the first stage of the QLoRA (Quantization for Low-Rank Adaptation) algorithm using CUDA. QLoRA is a technique that enables fine-tuning of large language models with reduced memory requirements by quantizing the model weights. This implementation focuses on the quantization aspect of QLoRA, where FP16 weights are quantized to INT4 representation.

   ### Key Concepts
   1. **Weight Quantization**:
      - Quantized FP16 weights to INT4 format to reduce memory requirements.
      - Used a block-wise quantization approach where each block of weights is quantized independently.
      - Computed scaling factors for each block to minimize quantization error.

   2. **CUDA Kernel for Quantization**:
      - Each thread processes a subset of weights within a block.
      - Computes the absolute maximum value within each block to determine the scaling factor.
      - Quantizes the weights by scaling and rounding to the nearest integer within the INT4 range [-8, 7].

   3. **CUDA Kernel for Dequantization**:
      - Converts the quantized INT4 weights back to FP16 format.
      - Uses the scaling factors to recover the approximate original values.
      - Enables verification of the quantization-dequantization process by measuring the RMSE.

   4. **Performance and Accuracy**:
      - Measured the Root Mean Square Error (RMSE) between the original weights and the dequantized weights.
      - The RMSE of 4.064262e-02 indicates a reasonable approximation, suitable for fine-tuning purposes.

   ### Results
   - **Dequantization RMSE**: 4.064262e-02

   ### Future Work
   - Implement the second stage of QLoRA: Low-Rank Adaptation (LoRA) for efficient fine-tuning.
   - Explore different quantization strategies to improve accuracy.
   - Benchmark the performance and memory savings of the complete QLoRA implementation.
   - Test the impact of quantization on model accuracy for various downstream tasks.

   ## Day 49
   **File:** `geglu_triton.py`

   ### Summary
   Implemented the GEGLU (Gaussian Error Linear Unit Gated) activation function using Triton. GEGLU is a variation of the GLU (Gated Linear Unit) that uses GELU activation instead of sigmoid for the gating mechanism. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **GEGLU Activation Function**:
      - The GEGLU function combines the GELU activation with a gating mechanism.
      - The formula for GEGLU is:
        
        $\text{GEGLU}(x) = \text{GELU}(x_1) \cdot x_2$
        
      - Here, $x_1$ and $x_2$ are the first and second halves of the input tensor, and GELU is the Gaussian Error Linear Unit activation function.

   2. **Triton Kernel for GEGLU**:
      - Each program processes a block of elements, loading data from global memory, performing the GEGLU computation, and storing the result back to global memory.
      - The kernel first computes the GELU activation for the first half of the input tensor, then multiplies it element-wise with the second half.

   3. **CPU Implementation**:
      - A CPU version of the GEGLU function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Performance**
     - Triton Execution Time: 0.009176 seconds
     - CPU Execution Time: 0.250023 seconds
     - Speedup: ~27.2x

   ### Future Work
   - Optimize the Triton kernel for better performance.
   ## Day 50
   **File:** `qlora_v2.cu`

   ### Summary
   Implemented the NF4 (NormalFloat 4-bit) quantization and dequantization algorithm for QLoRA using CUDA. This implementation focuses on efficiently quantizing floating-point weights to a 4-bit format with optimized representation, enabling significant memory reduction while preserving model quality for large language models.

   ### Key Concepts
   1. **NF4 Quantization**:
      - Uses a non-uniform quantization scheme with pre-defined quantization levels optimized for weight distributions in neural networks
      - Performs block-wise quantization where each block of weights is quantized independently
      - Computes a scaling factor for each block to preserve the dynamic range of values

   2. **CUDA Kernels**:
      - `quantize_nf4_kernel`: Converts floating-point values to 4-bit NF4 format by finding the closest match among predefined levels
      - `dequantize_nf4_kernel`: Converts the quantized NF4 values back to floating-point format
      - `double_quantize_scales_kernel`: Performs a second level of quantization on the scale factors to further reduce memory usage

   3. **Double Quantization**:
      - First quantizes weights to NF4 format with per-block scaling
      - Then quantizes the scale factors themselves to 4 bits, further reducing memory requirements
      - This technique is a key innovation in QLoRA that enables efficient fine-tuning of very large models

   4. **Packing Optimization**:
      - Packs two 4-bit indices into a single byte to optimize memory usage
      - Uses shared memory to improve efficiency in quantization and dequantization operations

   ### Results
   - **Quantization Validation**: Perfect match between CPU and GPU implementations
   - **Double Quantization**: Successfully quantized scale factors to 4-bit representation
   - **Dequantization RMSE**: 9.796087e-02, indicating a good approximation of original values
   - **Memory Reduction**: From 32-bit floating point to 4-bit, achieving 8x memory reduction

   ### Future Work
   - Implement the Low-Rank Adaptation (LoRA) part of QLoRA
   - Explore other quantization levels and schemes to improve accuracy

   ## Day 51
   **File:** `mish_triton.py`

   ### Summary
   Implemented the Mish activation function using Triton. Mish is a self-regularized non-monotonic activation function that combines properties of both ReLU and Swish activations. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Mish Activation Function**:
      - The Mish function is defined as:
        
        $\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))$
        
      - Where $\text{softplus}(x) = \ln(1 + e^x)$
      - It provides a smooth activation with better gradient properties than ReLU.

   2. **Triton Kernel for Mish**:
      - Each program processes a block of elements, loading data from global memory, computing the Mish activation, and storing the result back to global memory.
      - The kernel computes softplus, tanh, and their composition efficiently.

   3. **CPU Implementation**:
      - A CPU version of the Mish function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Performance**
      - Triton Execution Time: 0.004167 seconds
      - CPU Execution Time: 0.296648 seconds


   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other activation functions and their Triton implementations.

   ## Day 52
   **File:** `triton_selu.py`

   ### Summary
   Implemented the SELU (Scaled Exponential Linear Unit) activation function using Triton. SELU is a self-normalizing activation function that induces a convergence to zero mean and unit variance for normalized inputs. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **SELU Activation Function**:
      - The SELU function is defined as:
        
        $\text{SELU}(x) = \lambda \begin{cases} 
        x & \text{if } x > 0 \\ 
        \alpha (\exp(x) - 1) & \text{if } x \leq 0 
        \end{cases}$
        
      - Here, $\lambda$ and $\alpha$ are predefined constants that ensure the self-normalizing property of the activation function.
      - $\lambda = 1.0507$ and $\alpha = 1.6732$ are chosen specifically to maintain zero mean and unit variance.

   2. **Triton Kernel for SELU**:
      - Each program processes a block of elements, loading data from global memory, computing the SELU activation, and storing the result back to global memory.
      - The kernel efficiently handles the conditional computation based on whether the input is positive or negative.

   3. **CPU Implementation**:
      - A CPU version of the SELU function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Validation**: Results match! Triton implementation is correct.
   - **Performance**
     - Triton Execution Time: 0.005692 seconds
     - CPU Execution Time: 2.330934 seconds
     - Speedup: ~409x

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other self-normalizing activation functions and their Triton implementations.

   ## Day 53
   **File:** `LeakyReLU_triton.py`

   ### Summary
   Implemented the Leaky ReLU activation function using Triton. The Leaky ReLU function is a variant of ReLU that allows a small gradient for negative inputs, which helps prevent the "dying ReLU" problem. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Leaky ReLU Activation Function**:
      - The Leaky ReLU function is defined as:
        
        $$\text{Leaky ReLU}(x) = \begin{cases} 
        x & \text{if } x > 0 \\
        \alpha x & \text{if } x \leq 0 
        \end{cases}$$
        
      - Here, $\alpha$ is a small constant (typically 0.01) that determines the slope of the function for negative inputs.
      - Unlike standard ReLU which outputs zero for negative inputs, Leaky ReLU allows a small, non-zero gradient.

   2. **Triton Kernel for Leaky ReLU**:
      - Each program processes a block of elements, loading data from global memory, computing the Leaky ReLU activation, and storing the result back to global memory.
      - The kernel efficiently handles the conditional computation based on whether the input is positive or negative.

   3. **CPU Implementation**:
      - A CPU version of the Leaky ReLU function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Validation**: Results match! Triton implementation is correct.
   - **Performance**
     - Triton Execution Time: 0.004329 seconds
     - CPU Execution Time: 0.922624 seconds
     - Speedup: ~213x

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other variants of ReLU and their Triton implementations.
   - Investigate the impact of different alpha values on model performance.

   ## Day 54
   **File:** `triton_kl_divergence.py`

   ### Summary
   Implemented the KL Divergence (Kullback-Leibler Divergence) using Triton. KL Divergence is a measure of how one probability distribution diverges from a second, reference probability distribution. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **KL Divergence**:
      - A measure of how one probability distribution \( P \) diverges from a second, reference probability distribution \( Q \).
      - Defined as:
        
        $$D_{KL}(P || Q) = \sum_{i} P(i) \cdot \log\left(\frac{P(i)}{Q(i)}\right)$$
        
      - Lower KL divergence indicates that the distributions are more similar.

   2. **Triton Kernel for KL Divergence**:
      - Each program processes a block of elements, loading data from global memory, computing the KL divergence contribution, and storing the partial results back to global memory.
      - The kernel efficiently computes the logarithm and multiplication operations required for KL divergence.

   3. **CPU Implementation**:
      - A CPU version of the KL divergence function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Validation**: Results match! Triton implementation is correct.
   - **Performance**
     - Triton Execution Time: 0.015871 seconds
     - CPU Execution Time: 0.222043 seconds
     - Speedup: ~14x

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore applications of KL divergence in machine learning models, such as variational autoencoders and generative models.
   - Implement other divergence measures like Jensen-Shannon divergence using Triton.

   ## Day 55
   **File:** `triton_smooth_swiglu.py`

   ### Summary
   Implemented the Smooth SwiGLU (Swish Gated Linear Unit) activation function using Triton. The Smooth SwiGLU function combines the Swish activation with a gating mechanism and adds a smoothing sigmoid component. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Smooth SwiGLU Activation Function**:
      - The Smooth SwiGLU function combines the Swish activation with a gating mechanism.
      - The formula for Smooth SwiGLU is:
        
        $\text{Smooth SwiGLU}(x) = \text{Swish}(x_1) \cdot x_2$
        
      - Where $\text{Swish}(x) = x \cdot \text{sigmoid}(x)$
      - The input tensor is split into two halves, with the first half processed through Swish activation before being multiplied by the second half.

   2. **Triton Kernel for Smooth SwiGLU**:
      - Each program processes a block of elements, loading data from global memory, performing the Smooth SwiGLU computation, and storing the result back to global memory.
      - The kernel first computes the Swish activation for the first half of the input tensor, then multiplies it element-wise with the second half.

   3. **CPU Implementation**:
      - A CPU version of the Smooth SwiGLU function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Validation**: Results match! Triton implementation is correct.
   - **Performance**
     - Triton Execution Time: 0.011216 seconds
     - CPU Execution Time: 0.208424 seconds
     - Speedup: ~18.6x

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other activation functions and their Triton implementations.

   ## Day 56
   **File:** `triton_batchnorm.py`

   ### Summary
   Implemented batch normalization using Triton. Batch normalization is a technique used to improve the training of deep neural networks by normalizing layer inputs. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Batch Normalization**:
      - Normalizes the input by subtracting the mean and dividing by the standard deviation, followed by scaling and shifting.
      - The formula for batch normalization is:
        
        $$\text{BN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
        
      - Here, $\mu$ is the mean, $\sigma^2$ is the variance, $\epsilon$ is a small constant for numerical stability, $\gamma$ is the scale parameter, and $\beta$ is the shift parameter.

   2. **Triton Kernel for Batch Normalization**:
      - Each program processes a block of elements, loading data from global memory, performing the normalization, and storing the result back to global memory.
      - The kernel efficiently handles the normalization process in a single pass through the data.

   3. **CPU Implementation**:
      - A CPU version of the batch normalization function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   ### Results
   - **Validation**: Test passed! Triton and CPU results match.
   - **Performance**
     - Triton Time: 0.021808 sec
     - CPU Time: 0.554200 sec
     - Speedup: 25.41x

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Implement other normalization techniques using Triton.

   ## Day 57
   **File:** `triton_backward_batchnorm.py`

   ### Summary
   Implemented the backward pass for batch normalization using Triton. Batch normalization is a technique used to improve the training of deep neural networks, and its backward pass involves computing gradients with respect to the input, scale, and shift parameters. This implementation includes both forward and backward passes, with results validated against PyTorch's implementation.

   ### Key Concepts
   1. **Batch Normalization Backward Pass**:
      - Computes gradients for the input data, scale parameter (gamma), and shift parameter (beta).
      - The backward pass involves computing intermediate values like the gradients with respect to the normalized inputs, variance, and mean.

   2. **Triton Kernel for Backward Batch Normalization**:
      - Implements efficient parallel computation of gradients.
      - Uses atomic operations to correctly accumulate gradients for parameters across threads.

   3. **PyTorch Integration**:
      - Implemented as a PyTorch `autograd.Function` to seamlessly integrate with PyTorch's automatic differentiation system.
      - Allows direct comparison with PyTorch's native batch normalization implementation.

   4. **Performance and Accuracy**:
      - The implementation achieves high accuracy.
      - Uses relaxed tolerances for validation due to floating-point precision differences.

   ### Results

   - **Forward diff**: 4.76837158203125e-07
   - **dx diff**: 9.5367431640625e-07
   - **dgamma diff**: 5.7220458984375e-06
   - **dbeta diff**: 3.814697265625e-06
   - All tests passed with relaxed tolerances!

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Implement other normalization techniques and their backward passes using Triton.
   - Explore using higher precision arithmetic for improved accuracy.

## Day 58
   **File:** `triton_glu.py`

   ### Summary
   Implemented the Gated Linear Unit (GLU) activation function using Triton. GLU is a neural network layer that uses a gating mechanism to control information flow. This implementation includes both forward and backward passes with validation against PyTorch's implementation.

   ### Key Concepts
   1. **Gated Linear Unit (GLU)**:
      - The GLU function splits the input tensor into two parts along the feature dimension.
      - One part is transformed through a sigmoid function to create a gate.
      - The formula for GLU is:
        
        $$GLU(x) = x_1 \cdot \sigma(x_2)$$
        
      - Where $x_1$ and $x_2$ are the two halves of the input, and $\sigma$ is the sigmoid function.

   2. **Triton Kernel for GLU**:
      - Implements forward pass by splitting the input tensor and applying the GLU function.
      - Backward pass computes gradients for both parts of the input tensor.
      - Utilizes Triton's parallel processing capabilities for efficient computation.

   3. **PyTorch Integration**:
      - Implemented as a custom PyTorch function with autograd support.
      - Enables seamless integration with PyTorch models.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - The Triton implementation achieved a significant speedup over the CPU version.

   ### Results
   - **Validation**:
     - Max forward difference: 4.77e-07
     - Max backward difference: 4.77e-07
     - Test passed!

   - **Performance**:
     - CUDA Time: 0.0713 sec
     - CPU Time: 1.2998 sec
     - Speedup: 18.23x

   ### Future Work
   - Optimize the Triton kernel for even better performance.
   - Explore variants like Bilinear GLU for different applications.
   - Implement other gating mechanisms and compare their performance.

   ## Day 59
   **File:** `tanh_alpha.cu`

   ### Summary
   Implemented a parameterized hyperbolic tangent (tanh) function using CUDA. This implementation computes `tanh(alpha * x)` for each element in an array, where alpha is a scaling factor that controls the steepness of the tanh curve. Both CUDA and CPU implementations were created for performance comparison. Recent research in [Transformers without Normalization](https://arxiv.org/pdf/2503.10622) has shown that `tanh(alpha * x)` can effectively replace batch normalization layers in transformer architectures, offering similar performance while reducing computational complexity.

   ### Key Concepts
   1. **Parameterized tanh Function**:
      - The function computes `tanh(alpha * x)` where `alpha` controls the steepness of the curve.
      - Larger values of `alpha` make the transition between -1 and 1 sharper, which can be useful in neural networks to control how quickly the activation saturates.

   2. **CUDA Kernel Implementation**:
      - Each thread processes a single element of the input array.
      - The kernel uses the CUDA intrinsic function `tanhf()` for efficient computation.

   3. **Performance Comparison**:
      - Measured the execution time of both the CPU and GPU implementations.
      - Verified the correctness of results by comparing CPU and GPU outputs.

   ### Results
   - **Test Parameters**:
     - Array size: 1,048,576 elements (1M)
     - Alpha value: 2.0
   - **Performance**:
     - CPU time: 74,056 microseconds
     - GPU time: 230 microseconds
     - Speedup: 321.983x
   - **Validation**: PASSED

   ### Future Work
   - Explore other parameterized activation functions and their CUDA implementations.
   - Investigate the impact of different alpha values on neural network performance.
   - Implement batch processing for multiple alpha values simultaneously.


   ## Day 60
   **File:** `pos_enc.cu`

   ### Summary
   Implemented positional encoding using CUDA. Positional encoding is a crucial component in transformer architectures that helps the model understand the sequence order. This implementation includes both a CUDA kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Positional Encoding**:
      - Adds information about the position of tokens in a sequence to make the model aware of word order.
      - The formula for positional encoding is:
        
        $$PE_{(pos, 2i)} = \sin(pos \cdot 10000^{-2i/d_{model}})$$
        $$PE_{(pos, 2i+1)} = \cos(pos \cdot 10000^{-2i/d_{model}})$$
        
      - Here, $pos$ is the position in the sequence, $i$ is the dimension index, and $d_{model}$ is the model dimension.

   2. **CUDA Kernel for Positional Encoding**:
      - Each thread computes the positional encoding for a pair of dimensions (sin and cos values) at a specific position.
      - The kernel efficiently organizes computation using 2D thread blocks, mapping to positions and dimension pairs.

   3. **CPU Implementation**:
      - A CPU version of the positional encoding function is implemented for comparison with the GPU results.
      - This helps in verifying the correctness of the CUDA implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the CPU and GPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

   ### Results
   - **Test Parameters**:
     - Sequence length: 512
     - Model dimension: 256
     - Total elements: 131,072
   - **Performance**:
     - CPU time: 2,600 microseconds
     - GPU time: 290 microseconds
     - Speedup: 8.97x
   - **Validation**: PASSED

   ### Future Work
   - Optimize the CUDA kernel for better performance.
   - Explore variations of positional encoding like relative positional encoding.
   - Investigate the impact of different positional encoding schemes on transformer model performance.

   ## Day 61
   **File:** `glu.cu`

   ### Summary
   Implemented the Gated Linear Unit (GLU) activation function using CUDA. GLU is a neural network layer that uses a gating mechanism to control information flow. This implementation includes both a CUDA kernel and a CPU version for comparison.

   ### Key Concepts
   1. **Gated Linear Unit (GLU)**:
      - The GLU function splits the input tensor into two parts along the feature dimension.
      - One part is transformed through a sigmoid function to create a gate.
      - The formula for GLU is:
        
        $$GLU(x) = x_1 \cdot \sigma(x_2)$$
        
      - Where $x_1$ and $x_2$ are the two halves of the input, and $\sigma$ is the sigmoid function.

   2. **CUDA Kernel for GLU**:
      - Implements forward pass by splitting the input tensor and applying the GLU function.
      - Utilizes CUDA's parallel processing capabilities for efficient computation.

   3. **CPU Implementation**:
      - A CPU version of the GLU function is implemented for comparison with the GPU results.
      - This helps in verifying the correctness of the CUDA implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the CPU and GPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using the GPU.

   ### Results
   - **Performance**:
     - Total input elements: 1,048,576
     - Output size: 524,288
     - CPU time: 6,896 microseconds
     - GPU time: 167 microseconds
     - Speedup: 41.2934x
   - **Validation**: Test PASSED

   ### Future Work
   - Optimize the CUDA kernel for better performance.
   - Explore other gating mechanisms and their CUDA implementations.
   - Validate the implementation with larger datasets and different configurations.

   ## Day 62
   **File:** `triton_relu2.py`

   ### Summary
   Implemented the ReLU² (Squared ReLU) activation function using Triton. ReLU² squares the positive inputs while zeroing out negative values. This implementation includes both a Triton kernel and a CPU version for comparison.

   ### Key Concepts
   1. **ReLU² Activation Function**:
      - The ReLU² function is defined as:
        
        $$\text{ReLU}^2(x) = \begin{cases} 
        x^2 & \text{if } x > 0 \\ 
        0 & \text{if } x \leq 0 
        \end{cases}$$
        
      - It introduces non-linearity while maintaining a steeper gradient for positive inputs compared to standard ReLU.

   2. **Triton Kernel for ReLU²**:
      - Each program processes a block of elements, loading data from global memory, performing the ReLU² computation, and storing the result back to global memory.
      - The kernel evaluates whether the input is positive, and if so, squares it; otherwise, it outputs zero.

   3. **CPU Implementation**:
      - A CPU version of the ReLU² function is implemented for comparison with the Triton results.
      - This helps in verifying the correctness of the Triton implementation.

   4. **Performance Comparison**:
      - Measured the execution time of both the Triton and CPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.


   ### Future Work
   - Investigate the impact of ReLU² on model convergence and accuracy.
   - Optimize the Triton kernel for better performance.
   - Explore other activation functions and their Triton implementations.

   ## Day 63
   **File:** `triton_mse.py`

   ### Summary
   Implemented the Mean Squared Error (MSE) loss function using Triton. The MSE loss is a common metric used in regression tasks to measure the average squared difference between predicted and target values. This implementation includes both a Triton kernel and CPU/PyTorch versions for comparison.

   ### Key Concepts
   1. **MSE Loss Function**:
      - The MSE loss is defined as:
        
        $$\text{MSE}(y_{\text{pred}}, y_{\text{true}}) = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{pred},i} - y_{\text{true},i})^2$$
        
      - It measures the average squared difference between predictions and targets.

   2. **Triton Kernel for MSE**:
      - Each program processes a block of elements, loading data from global memory, computing the squared differences, and accumulating the results.
      - The kernel uses atomic operations to ensure correct accumulation of partial results.

   3. **Performance Comparison**:
      - Measured the execution time of the CPU, PyTorch GPU, and Triton GPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   4. **Validation**:
      - Verified the correctness of the Triton implementation by comparing its results with the CPU and PyTorch GPU implementations.
      - All implementations matched within a tolerance of \(1 \times 10^{-5}\).

   ### Results
   - **Validation Output**:
     - CPU result: 2.0377094745635986
     - Triton GPU result: 2.0377094745635986
     - PyTorch GPU result: 2.0377092361450195
     - Difference between CPU and Triton: 0.0
     - Difference between CPU and PyTorch: \(2.384185791015625 \times 10^{-7}\)
     - **Test passed!** All implementations match within tolerance.

   - **Performance Results**:
   ```markdown
   | Size       | CPU (ms) | PyTorch GPU (ms) | Triton GPU (ms) | Speedup (Triton vs CPU) |
   |------------|----------|------------------|-----------------|-------------------------|
   | 1,000      | 0.028    | 0.027            | 0.064           | 0.4x                   |
   | 10,000     | 0.028    | 0.027            | 0.056           | 0.5x                   |
   | 100,000    | 0.129    | 0.035            | 0.055           | 2.3x                   |
   | 1,000,000  | 4.372    | 0.075            | 0.057           | 76.7x                  |
   | 10,000,000 | 54.530   | 0.638            | 0.292           | 186.6x                 |
   ```

   ### Analysis
   - **Accuracy**:
     - The Triton implementation produces results that are identical to the CPU implementation and very close to the PyTorch GPU implementation, with negligible differences due to floating-point precision.

   - **Performance**:
     - For smaller input sizes, the Triton implementation is slower than the CPU and PyTorch GPU implementations due to kernel launch overhead.
     - For larger input sizes, the Triton implementation achieves significant speedups, with up to **186.6x** speedup over the CPU implementation for \(10^7\) elements.

   ### Future Work
   - Optimize the Triton kernel for smaller input sizes to reduce overhead.
   - Explore other loss functions and their Triton implementations.
   - Validate the implementation with larger datasets and different data types.
   - Investigate the impact of block size and grid configuration on performance.
   - Compare performance with other GPU-accelerated libraries for MSE loss computation.

   ## Day 64
   **File:** `triton_triplet_loss.py`

   ### Summary
   Implemented the Triplet Loss function using Triton. Triplet Loss is commonly used in metric learning tasks to ensure that similar samples are closer in the embedding space while dissimilar samples are farther apart. This implementation includes both a Triton kernel and CPU/PyTorch versions for comparison.

   ### Key Concepts
   1. **Triplet Loss Function**:
      - The Triplet Loss is defined as:
        
        $$\text{Loss} = \max(0, ||a - p||_2 - ||a - n||_2 + \text{margin})$$
        
      - Here:
        - \(a\): Anchor embedding
        - \(p\): Positive embedding (similar to anchor)
        - \(n\): Negative embedding (dissimilar to anchor)
        - \($\text{margin}$\): A hyperparameter that defines the minimum distance between positive and negative pairs.

   2. **Triton Kernel for Triplet Loss**:
      - Each program processes a block of triplets, loading data from global memory, computing distances, and applying the loss formula.
      - The kernel uses efficient memory access patterns and parallelism to compute the loss for large batches of triplets.

   3. **Performance Comparison**:
      - Measured the execution time of the CPU, PyTorch GPU, and Triton GPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   4. **Validation**:
      - Verified the correctness of the Triton implementation by comparing its results with the CPU and PyTorch GPU implementations.
      - All implementations matched within a tolerance of \($1 \times 10^{-7}$\).

   ### Results
   - **Validation Output**:
     - CPU result: 1.16278874874115
     - Triton GPU result: 1.1627888679504395
     - PyTorch GPU result: 1.1627886295318604
     - Difference between CPU and Triton: \(1.1920928955078125 $\times$ $10^{-7}$\)
     - Difference between CPU and PyTorch: \(1.1920928955078125 $\times$ $10^{-7}\$)
     - **Test passed!** All implementations match within tolerance.

   ### Future Work
   - Optimize the Triton kernel for better performance.
   - Explore other loss functions and their Triton implementations.
   - Validate the implementation with larger datasets and different embedding dimensions.
   - Investigate the impact of different margin values on model performance.
   - Compare performance with other GPU-accelerated libraries for Triplet Loss computation.


   ## Day 65
   **File:** `triton_r2.py`

   ### Summary
   Implemented the R-squared (coefficient of determination) metric using Triton. R-squared is a statistical measure used to evaluate the goodness of fit of a regression model. This implementation includes both a Triton kernel and CPU/PyTorch versions for comparison.

   ### Key Concepts
   1. **R-squared Formula**:
      - The R-squared metric is defined as:
        
        $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
        
      - Where:
        - \($\text{SS}_{\text{res}}$\): Residual sum of squares
        - \($\text{SS}_{\text{tot}}$\): Total sum of squares

   2. **Triton Kernel for R-squared**:
      - Each program processes a block of elements, loading data from global memory, computing the residuals and total sums, and accumulating the results.
      - The kernel uses atomic operations to ensure correct accumulation of partial results.

   3. **Performance Comparison**:
      - Measured the execution time of the CPU, PyTorch GPU, and Triton GPU implementations.
      - Compared the results to ensure they match and evaluated the speedup achieved by using Triton.

   4. **Validation**:
      - Verified the correctness of the Triton implementation by comparing its results with the CPU and Sklearn-style implementations.
      - All implementations matched within a tolerance of \($1 \times 10^{-7}$\).

   ### Results
   - **Validation Output**:
     - CPU result: -1.0076630115509033
     - Triton GPU result: -1.0076625347137451
     - Sklearn-style result: -1.0076630115509033
     - Difference between CPU and Triton: \($4.76837158203125 \times 10^{-7}$\)
     - Difference between CPU and Sklearn-style: 0.0
     - **Test passed!** All implementations match within tolerance.
   ### Performance Results

   | Size       | CPU (ms) | PyTorch GPU (ms) | Triton GPU (ms) | Speedup (Triton vs CPU) |
   |------------|----------|------------------|-----------------|-------------------------|
   | 1,000      | 0.062    | 0.105            | 0.167           | 0.4x                   |
   | 10,000     | 0.087    | 0.197            | 0.265           | 0.3x                   |
   | 100,000    | 0.271    | 0.172            | 0.286           | 0.9x                   |
   | 1,000,000  | 10.870   | 0.251            | 0.205           | 53.1x                  |
   | 10,000,000 | 142.884  | 1.964            | 0.443           | 322.4x                 |

   ### Analysis
   - **Accuracy**:
     - The Triton implementation produces results that are nearly identical to the CPU and Sklearn-style implementations, with negligible differences due to floating-point precision.

   - **Performance**:
     - For smaller input sizes, the Triton implementation is slower than the CPU and PyTorch GPU implementations due to kernel launch overhead.
     - For larger input sizes, the Triton implementation achieves significant speedups, with up to **322.4x** speedup over the CPU implementation for \(10^7\) elements.

   ### Future Work
   - Optimize the Triton kernel for smaller input sizes to reduce overhead.
   - Investigate the numerical issue observed at smaller sizes.
   - Explore other regression metrics and their Triton implementations.
   - Validate the implementation with larger datasets and different data types.
   - Investigate the impact of block size and grid configuration on performance.