import torch
import triton
import triton.language as tl
import time


# Triton Kernel for Smooth SwiGLU
@triton.jit
def smooth_swiglu_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Index of the current program (thread block)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements // 2)  # Mask to avoid out-of-bounds access

    # Load data from input tensor
    x1 = tl.load(x_ptr + offsets, mask=mask)  # First half of the input
    x2 = tl.load(
        x_ptr + (n_elements // 2) + offsets, mask=mask
    )  # Second half of the input
    # Compute Swish(x1): x1 * sigmoid(x1)
    sigmoid_x1 = 1.0 / (1.0 + tl.exp(-x1))  # Sigmoid function
    swish_x1 = x1 * sigmoid_x1  # Swish(x1) = x1 * sigmoid(x1)
    # Compute Smooth SwiGLU: y = Swish(x1) * x2
    y = swish_x1 * x2
    # Store result to output tensor
    tl.store(y_ptr + offsets, y, mask=mask)


# Wrapper function to call the Triton kernel
def smooth_swiglu_triton(x: torch.Tensor):
    n_elements = x.numel()  # Total number of elements in the input tensor
    y = torch.empty(
        n_elements // 2, device=x.device, dtype=x.dtype
    )  # Output tensor (half the size)
    grid = lambda meta: (
        triton.cdiv(n_elements // 2, meta["BLOCK_SIZE"]),
    )  # Grid function
    smooth_swiglu_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)  # Launch kernel
    return y


# CPU Implementation of Smooth SwiGLU for Comparison
def smooth_swiglu_cpu(x: torch.Tensor):
    half_size = x.size(0) // 2
    x1 = x[:half_size]
    x2 = x[half_size:]
    # Compute Swish(x1): x1 * sigmoid(x1)
    sigmoid_x1 = torch.sigmoid(x1)  # Sigmoid function
    swish_x1 = x1 * sigmoid_x1  # Swish(x1) = x1 * sigmoid(x1)
    # Compute Smooth SwiGLU: y = Swish(x1) * x2
    return swish_x1 * x2


# Test Function
def test_smooth_swiglu():
    # Create a random tensor
    size = 1 << 20  # 1 million elements
    x = torch.rand(size, device="cuda") * 2 - 1  # Values between -1 and 1

    # Compute results using Triton and CPU
    y_triton = smooth_swiglu_triton(x)
    y_cpu = smooth_swiglu_cpu(x.cpu()).cuda()  # Move to GPU for comparison

    # Validate correctness
    assert torch.allclose(y_triton, y_cpu, rtol=1e-5), (
        "Triton and CPU results do not match!"
    )
    print("Results match! Triton implementation is correct.")


# Performance Comparison
def performance_comparison():
    size = 1 << 20  # 1 million elements
    x = torch.rand(size, device="cuda") * 2 - 1  # Values between -1 and 1

    # Warm-up (to avoid initial overhead)
    for _ in range(10):
        smooth_swiglu_triton(x)

    # Benchmark Triton
    start_time = time.time()
    for _ in range(100):
        smooth_swiglu_triton(x)
    triton_time = time.time() - start_time
    print(f"Triton Execution Time: {triton_time:.6f} seconds")

    # Benchmark CPU
    x_cpu = x.cpu()
    start_time = time.time()
    for _ in range(100):
        smooth_swiglu_cpu(x_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU Execution Time: {cpu_time:.6f} seconds")


# Run tests and performance comparison
if __name__ == "__main__":
    test_smooth_swiglu()
    performance_comparison()
