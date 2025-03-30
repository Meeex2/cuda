import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time
import numpy as np


# ReLU2 function: relu(x)^2
def relu2_cpu(x):
    return torch.clamp(x, min=0) ** 2


@triton.jit
def relu2_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    pid = tl.program_id(axis=0)  # We use 1D launch grid so axis is 0
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # ReLU2 computation
    zero = tl.zeros(x.shape, x.dtype)
    y = tl.where(x > zero, x, zero)
    y = y * y

    # Write back result
    tl.store(y_ptr + offsets, y, mask=mask)


def relu2_gpu(x):
    # Allocate output
    y = torch.empty_like(x)

    # The SPMD launch grid denotes the number of kernel instances in parallel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    relu2_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=1024,  # Can be tuned for optimal performance
    )

    return y


# Benchmarking function
def benchmark(fn, x, num_warmups=10, num_iters=100):
    # Warmup
    for _ in range(num_warmups):
        fn(x)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        fn(x)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / num_iters
    return elapsed_ms


# Test function
def test_relu2():
    # Test data
    x = torch.randn(10000, device="cuda")

    # Compute results
    y_cpu = relu2_cpu(x.cpu()).cuda()
    y_gpu = relu2_gpu(x)

    # Check correctness
    assert torch.allclose(y_cpu, y_gpu, atol=1e-6), "CPU and GPU implementations differ"
    print("Test passed! CPU and GPU implementations match.")


# Performance comparison
def compare_performance():
    sizes = [10**i for i in range(3, 8)]  # 1K to 100M elements
    cpu_times = []
    gpu_times = []
    triton_times = []

    for size in sizes:
        print(f"Benchmarking size: {size}")
        x = torch.randn(size, device="cuda")

        # CPU benchmark
        x_cpu = x.cpu()
        start = time.time()
        for _ in range(10):
            relu2_cpu(x_cpu)
        cpu_time = (time.time() - start) * 1000 / 10
        cpu_times.append(cpu_time)

        # GPU PyTorch benchmark
        torch_relu2 = lambda x: torch.clamp(x, min=0).pow(2)
        gpu_time = benchmark(torch_relu2, x)
        gpu_times.append(gpu_time)

        # Triton benchmark
        triton_time = benchmark(relu2_gpu, x)
        triton_times.append(triton_time)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, label="CPU")
    plt.plot(sizes, gpu_times, label="GPU (PyTorch)")
    plt.plot(sizes, triton_times, label="GPU (Triton)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Input size (elements)")
    plt.ylabel("Time (ms)")
    plt.title("ReLU2 Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("relu2_performance.png")
    plt.show()

    # Print results
    print("\nPerformance Results:")
    print(
        f"{'Size':>10} {'CPU (ms)':>10} {'PyTorch GPU (ms)':>15} {'Triton GPU (ms)':>15} {'Speedup (Triton vs CPU)':>20}"
    )
    for i, size in enumerate(sizes):
        speedup = cpu_times[i] / triton_times[i]
        print(
            f"{size:10} {cpu_times[i]:10.3f} {gpu_times[i]:15.3f} {triton_times[i]:15.3f} {speedup:20.1f}x"
        )


if __name__ == "__main__":
    test_relu2()
    compare_performance()
