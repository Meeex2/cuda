import torch
import triton
import triton.language as tl
import time
import numpy as np


# Triton Kernel for Batch Normalization
@triton.jit
def batch_norm_kernel(
    x_ptr,  # Input tensor
    y_ptr,  # Output tensor
    mean_ptr,  # Mean of input
    var_ptr,  # Variance of input
    gamma_ptr,  # Scale parameter
    beta_ptr,  # Shift parameter
    eps,  # Small constant for numerical stability
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr)
    var = tl.load(var_ptr)
    gamma = tl.load(gamma_ptr)
    beta = tl.load(beta_ptr)

    # Normalize: (x - mean) / sqrt(var + eps)
    normalized = (x - mean) / tl.sqrt(var + eps)
    # Scale and shift
    y = gamma * normalized + beta

    tl.store(y_ptr + offsets, y, mask=mask)


def batch_norm_triton(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
):
    # Compute mean and variance
    mean = x.mean()
    var = x.var(unbiased=False)

    # Allocate output
    y = torch.empty_like(x)
    n_elements = x.numel()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    batch_norm_kernel[grid](
        x, y, mean, var, gamma, beta, eps, n_elements, BLOCK_SIZE=1024
    )
    return y


# CPU Implementation for comparison
def batch_norm_cpu(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
):
    mean = x.mean()
    var = x.var(unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


# Test Function
def test_batch_norm():
    # Create random input
    size = 1 << 20  # 1M elements
    x = torch.rand(size, device="cuda") * 2 - 1  # Values between -1 and 1
    gamma = torch.tensor(1.0, device="cuda")
    beta = torch.tensor(0.0, device="cuda")
    eps = 1e-5

    # Compute results
    y_triton = batch_norm_triton(x, gamma, beta, eps)
    y_cpu = batch_norm_cpu(x.cpu(), gamma.cpu(), beta.cpu(), eps).cuda()

    # Validate correctness
    assert torch.allclose(y_triton, y_cpu, rtol=1e-4), "Results don't match!"
    print("Test passed! Triton and CPU results match.")


# Performance Comparison
def performance_comparison():
    size = 1 << 20  # 1M elements
    x = torch.rand(size, device="cuda") * 2 - 1
    gamma = torch.tensor(1.0, device="cuda")
    beta = torch.tensor(0.0, device="cuda")
    eps = 1e-5

    # Warm-up
    for _ in range(10):
        batch_norm_triton(x, gamma, beta, eps)

    # Benchmark Triton
    start = time.time()
    for _ in range(100):
        batch_norm_triton(x, gamma, beta, eps)
    triton_time = time.time() - start
    print(f"Triton Time: {triton_time:.6f} sec")

    # Benchmark CPU
    x_cpu = x.cpu()
    gamma_cpu = gamma.cpu()
    beta_cpu = beta.cpu()
    start = time.time()
    for _ in range(100):
        batch_norm_cpu(x_cpu, gamma_cpu, beta_cpu, eps)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.6f} sec")
    print(f"Speedup: {cpu_time / triton_time:.2f}x")


if __name__ == "__main__":
    test_batch_norm()
    performance_comparison()
