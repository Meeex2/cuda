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
