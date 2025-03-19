import torch
import triton
import triton.language as tl
import time
import math


# Triton Kernel for Mish
@triton.jit
def mish_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Index of the current program (thread block)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to avoid out-of-bounds access

    # Load data from input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute softplus(x) = ln(1 + e^x)
    softplus_x = tl.log(1.0 + tl.exp(x))
    # Compute tanh(softplus(x)) manually: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    exp_2x = tl.exp(2.0 * softplus_x)
    tanh_softplus_x = (exp_2x - 1.0) / (exp_2x + 1.0)
    # Compute Mish: y = x * tanh(softplus(x))
    y = x * tanh_softplus_x
    # Store result to output tensor
    tl.store(y_ptr + offsets, y, mask=mask)


# Wrapper function to call the Triton kernel
def mish_triton(x: torch.Tensor):
    y = torch.empty_like(x)  # Output tensor
    n_elements = x.numel()  # Total number of elements
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # Grid function
    mish_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)  # Launch kernel
    return y


# CPU Implementation of Mish for Comparison
def mish_cpu(x: torch.Tensor):
    softplus_x = torch.log(1.0 + torch.exp(x))  # softplus(x) = ln(1 + e^x)
    tanh_softplus_x = torch.tanh(softplus_x)  # tanh(softplus(x))
    return x * tanh_softplus_x  # Mish(x) = x * tanh(softplus(x))


# Test Function
def test_mish():
    # Create a random tensor
    size = 1 << 20  # 1 million elements
    x = torch.rand(size, device="cuda") * 2 - 1  # Values between -1 and 1

    # Compute results using Triton and CPU
    y_triton = mish_triton(x)
    y_cpu = mish_cpu(x.cpu()).cuda()  # Move to GPU for comparison

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
        mish_triton(x)

    # Benchmark Triton
    start_time = time.time()
    for _ in range(100):
        mish_triton(x)
    triton_time = time.time() - start_time
    print(f"Triton Execution Time: {triton_time:.6f} seconds")

    # Benchmark CPU
    x_cpu = x.cpu()
    start_time = time.time()
    for _ in range(100):
        mish_cpu(x_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU Execution Time: {cpu_time:.6f} seconds")


# Run tests and performance comparison
if __name__ == "__main__":
    test_mish()
    performance_comparison()
