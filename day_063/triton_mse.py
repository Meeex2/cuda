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


# Performance comparison
def compare_performance():
    sizes = [10**i for i in range(3, 8)]  # 1K to 100M elements
    cpu_times = []
    pytorch_times = []
    triton_times = []

    for size in sizes:
        print(f"\nBenchmarking size: {size}")
        pred = torch.randn(size, device="cuda")
        target = torch.randn(size, device="cuda")

        # First verify correctness at this size
        cpu_val = mse_loss_cpu(pred.cpu(), target.cpu())
        triton_val = mse_loss_gpu(pred, target)
        if not torch.allclose(cpu_val.cuda(), triton_val, atol=1e-5):
            print(f"Warning: Potential numerical issue at size {size}")
            print(f"CPU: {cpu_val.item()}, Triton: {triton_val.item()}")

        # CPU benchmark
        pred_cpu = pred.cpu()
        target_cpu = target.cpu()
        start = time.time()
        for _ in range(10):
            mse_loss_cpu(pred_cpu, target_cpu)
        cpu_time = (time.time() - start) * 1000 / 10
        cpu_times.append(cpu_time)

        # PyTorch GPU benchmark
        pytorch_time = benchmark(torch.nn.functional.mse_loss, pred, target)
        pytorch_times.append(pytorch_time)

        # Triton benchmark
        triton_time = benchmark(mse_loss_gpu, pred, target)
        triton_times.append(triton_time)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, label="CPU")
    plt.plot(sizes, pytorch_times, label="GPU (PyTorch)")
    plt.plot(sizes, triton_times, label="GPU (Triton)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Input size (elements)")
    plt.ylabel("Time (ms)")
    plt.title("MSE Loss Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("mse_loss_performance.png")
    plt.show()

    # Print results
    print("\nPerformance Results:")
    print(
        f"{'Size':>10} {'CPU (ms)':>10} {'PyTorch GPU (ms)':>15} {'Triton GPU (ms)':>15} {'Speedup (Triton vs CPU)':>20}"
    )
    for i, size in enumerate(sizes):
        speedup = cpu_times[i] / triton_times[i]
        print(
            f"{size:10} {cpu_times[i]:10.3f} {pytorch_times[i]:15.3f} {triton_times[i]:15.3f} {speedup:20.1f}x"
        )


if __name__ == "__main__":
    test_mse_loss()
    compare_performance()
