import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time
import numpy as np


# MSE loss CPU implementation
def mse_loss_cpu(predictions, targets):
    return torch.mean((predictions - targets) ** 2)


@triton.jit
def mse_loss_kernel(
    pred_ptr,  # Pointer to predictions tensor
    target_ptr,  # Pointer to targets tensor
    output_ptr,  # Pointer to output (single value)
    n_elements,  # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    pid = tl.program_id(axis=0)  # We use 1D launch grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask to guard memory operations
    mask = offsets < n_elements

    # Load data
    pred = tl.load(pred_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)

    # Compute squared error
    diff = pred - target
    squared_error = diff * diff

    # Calculate the mean for this block
    block_mean = tl.sum(squared_error, axis=0) / n_elements

    # Atomic add to global output
    tl.atomic_add(output_ptr, block_mean)


def mse_loss_gpu(predictions, targets):
    # Allocate output (single value)
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)

    # The SPMD launch grid
    n_elements = predictions.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    mse_loss_kernel[grid](
        predictions,
        targets,
        output,
        n_elements,
        BLOCK_SIZE=1024,  # Can be tuned for optimal performance
    )

    return output


# Benchmarking function
def benchmark(fn, pred, target, num_warmups=10, num_iters=100):
    # Warmup
    for _ in range(num_warmups):
        fn(pred, target)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        fn(pred, target)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / num_iters
    return elapsed_ms


# Test function with more detailed error reporting
def test_mse_loss():
    # Test data - using smaller size for easier debugging
    size = 10000
    pred = torch.randn(size, device="cuda")
    target = torch.randn(size, device="cuda")

    # Compute results
    cpu_result = mse_loss_cpu(pred.cpu(), target.cpu()).cuda()
    gpu_result = mse_loss_gpu(pred, target)
    pytorch_result = torch.nn.functional.mse_loss(pred, target)

    # Print values for debugging
    print(f"CPU result: {cpu_result.item()}")
    print(f"Triton GPU result: {gpu_result.item()}")
    print(f"PyTorch GPU result: {pytorch_result.item()}")

    # Calculate absolute differences
    diff_triton = torch.abs(cpu_result - gpu_result).item()
    diff_pytorch = torch.abs(cpu_result - pytorch_result).item()
    print(f"Difference between CPU and Triton: {diff_triton}")
    print(f"Difference between CPU and PyTorch: {diff_pytorch}")

    # Check correctness with more relaxed tolerance
    atol = 1e-5  # Increased tolerance for floating point differences
    assert torch.allclose(cpu_result, gpu_result, atol=atol), (
        f"CPU and Triton GPU implementations differ by {diff_triton} (beyond tolerance {atol})"
    )
    assert torch.allclose(cpu_result, pytorch_result, atol=atol), (
        f"CPU and PyTorch GPU implementations differ by {diff_pytorch} (beyond tolerance {atol})"
    )
    print("Test passed! All implementations match within tolerance.")
