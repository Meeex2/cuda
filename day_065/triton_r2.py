import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time
import numpy as np


# R-squared CPU implementation
def r2_score_cpu(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))  # Small epsilon to avoid division by zero


@triton.jit
def r2_score_kernel(
    y_true_ptr,  # Pointer to true values
    y_pred_ptr,  # Pointer to predicted values
    mean_ptr,  # Pointer to mean value (single element)
    ss_res_ptr,  # Pointer to residual sum of squares (single element)
    ss_tot_ptr,  # Pointer to total sum of squares (single element)
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Elements per block
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    y_true = tl.load(y_true_ptr + offsets, mask=mask)
    y_pred = tl.load(y_pred_ptr + offsets, mask=mask)

    # Compute mean, ss_res, and ss_tot in parallel
    y_mean = tl.load(mean_ptr)
    ss_res = tl.sum((y_true - y_pred) * (y_true - y_pred))
    ss_tot = tl.sum((y_true - y_mean) * (y_true - y_mean))

    # Atomic add to global memory
    tl.atomic_add(ss_res_ptr, ss_res)
    tl.atomic_add(ss_tot_ptr, ss_tot)


def r2_score_gpu(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    n_elements = y_true.numel()

    # Allocate intermediate results
    mean = torch.mean(y_true)
    ss_res = torch.zeros(1, device=y_true.device)
    ss_tot = torch.zeros(1, device=y_true.device)

    # Kernel configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    r2_score_kernel[grid](
        y_true,
        y_pred,
        mean,
        ss_res,
        ss_tot,
        n_elements,
        BLOCK_SIZE=1024,
    )

    # Compute final R-squared score
    return 1 - (ss_res / (ss_tot + 1e-8))


# Benchmarking function
def benchmark(fn, y_true, y_pred, num_warmups=10, num_iters=100):
    # Warmup
    for _ in range(num_warmups):
        fn(y_true, y_pred)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        fn(y_true, y_pred)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / num_iters
    return elapsed_ms


# Test function with detailed error reporting
def test_r2_score():
    # Test data
    torch.manual_seed(42)
    size = 100000
    y_true = torch.randn(size, device="cuda")
    y_pred = torch.randn(size, device="cuda")

    # Compute results
    cpu_result = r2_score_cpu(y_true.cpu(), y_pred.cpu()).cuda()
    gpu_result = r2_score_gpu(y_true, y_pred)
    sklearn_result = torch.tensor(
        1
        - (
            torch.sum((y_true.cpu() - y_pred.cpu()) ** 2)
            / torch.sum((y_true.cpu() - torch.mean(y_true.cpu())) ** 2)
        ),
        device="cuda",
    )

    # Print values for debugging
    print(f"CPU result: {cpu_result.item()}")
    print(f"Triton GPU result: {gpu_result.item()}")
    print(f"Sklearn-style result: {sklearn_result.item()}")

    # Calculate absolute differences
    diff_triton = torch.abs(cpu_result - gpu_result).item()
    diff_sklearn = torch.abs(cpu_result - sklearn_result).item()
    print(f"Difference between CPU and Triton: {diff_triton}")
    print(f"Difference between CPU and Sklearn-style: {diff_sklearn}")

    # Check correctness with tolerance
    atol = 1e-5
    assert torch.allclose(cpu_result, gpu_result, atol=atol), (
        f"CPU and Triton GPU implementations differ by {diff_triton} (beyond tolerance {atol})"
    )
    assert torch.allclose(cpu_result, sklearn_result, atol=atol), (
        f"CPU and Sklearn-style implementations differ by {diff_sklearn} (beyond tolerance {atol})"
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
        y_true = torch.randn(size, device="cuda")
        y_pred = torch.randn(size, device="cuda")

        # Verify correctness first
        cpu_val = r2_score_cpu(y_true.cpu(), y_pred.cpu())
        triton_val = r2_score_gpu(y_true, y_pred)
        if not torch.allclose(cpu_val.cuda(), triton_val, atol=1e-5):
            print(f"Warning: Potential numerical issue at size {size}")
            print(f"CPU: {cpu_val.item()}, Triton: {triton_val.item()}")

        # CPU benchmark
        y_true_cpu = y_true.cpu()
        y_pred_cpu = y_pred.cpu()
        start = time.time()
        for _ in range(10):
            r2_score_cpu(y_true_cpu, y_pred_cpu)
        cpu_time = (time.time() - start) * 1000 / 10
        cpu_times.append(cpu_time)

        # PyTorch benchmark (manual implementation)
        def pytorch_r2(y_true, y_pred):
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))

        pytorch_time = benchmark(pytorch_r2, y_true, y_pred)
        pytorch_times.append(pytorch_time)

        # Triton benchmark
        triton_time = benchmark(r2_score_gpu, y_true, y_pred)
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
    plt.title("R-squared Score Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("r2_score_performance.png")
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
    test_r2_score()
    compare_performance()
