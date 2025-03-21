import torch
import triton
import triton.language as tl
import time

# Triton Kernel for Leaky ReLU
@triton.jit
def leaky_relu_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
    alpha: tl.constexpr,  # Negative slope for Leaky ReLU
):
    pid = tl.program_id(axis=0)  # Index of the current program (thread block)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to avoid out-of-bounds access

    # Load data from input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute Leaky ReLU: y = x if x > 0 else alpha * x
    y = tl.where(x > 0, x, alpha * x)
    # Store result to output tensor
    tl.store(y_ptr + offsets, y, mask=mask)


# Wrapper function to call the Triton kernel
def leaky_relu_triton(x: torch.Tensor, alpha: float = 0.01):
    y = torch.empty_like(x)  # Output tensor
    n_elements = x.numel()  # Total number of elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)  # Grid function
    leaky_relu_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024, alpha=alpha)  # Launch kernel
    return y


# CPU Implementation of Leaky ReLU for Comparison
def leaky_relu_cpu(x: torch.Tensor, alpha: float = 0.01):
    return torch.where(x > 0, x, alpha * x)


# Test Function
def test_leaky_relu():
    # Create a random tensor
    size = 1 << 20  # 1 million elements
    x = torch.rand(size, device='cuda') * 2 - 1  # Values between -1 and 1

    # Compute results using Triton and CPU
    alpha = 0.01  # Negative slope
    y_triton = leaky_relu_triton(x, alpha)
    y_cpu = leaky_relu_cpu(x.cpu(), alpha).cuda()  # Move to GPU for comparison


    # Validate correctness
    assert torch.allclose(y_triton, y_cpu, rtol=1e-5), "Triton and CPU results do not match!"
    print("Results match! Triton implementation is correct.")


# Performance Comparison
def performance_comparison():
    size = 1 << 20  # 1 million elements
    x = torch.rand(size, device='cuda') * 2 - 1  # Values between -1 and 1

    # Warm-up (to avoid initial overhead)
    for _ in range(10):
        leaky_relu_triton(x)

    # Benchmark Triton
    start_time = time.time()
    for _ in range(100):
        leaky_relu_triton(x)
    triton_time = time.time() - start_time
    print(f"Triton Execution Time: {triton_time:.6f} seconds")

    # Benchmark CPU
    x_cpu = x.cpu()
    start_time = time.time()
    for _ in range(100):
        leaky_relu_cpu(x_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU Execution Time: {cpu_time:.6f} seconds")


# Run tests and performance comparison
if __name__ == "__main__":
    test_leaky_relu()
    performance_comparison()