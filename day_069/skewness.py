import torch
import triton
import triton.language as tl
import time


@triton.jit
def skewness_kernel(
    x_ptr,  # Input data pointer
    mean,  # Mean value (loaded once)
    n_elements,  # Number of elements
    m2_ptr,  # Second moment accumulator
    m3_ptr,  # Third moment accumulator
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Center data and compute moments
    x_centered = x - tl.load(mean)
    x2 = x_centered * x_centered
    x3 = x2 * x_centered

    # Accumulate moments
    tl.atomic_add(m2_ptr, tl.sum(x2))
    tl.atomic_add(m3_ptr, tl.sum(x3))


def skewness_gpu(x):
    n = x.numel()

    # Compute mean on CPU (more numerically stable)
    mean_cpu = x.mean().item()
    mean = torch.tensor([mean_cpu], device=x.device)

    # Allocate moment accumulators
    m2 = torch.zeros(1, device=x.device)
    m3 = torch.zeros(1, device=x.device)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    skewness_kernel[grid](x, mean, n, m2, m3, BLOCK_SIZE=BLOCK_SIZE)

    # Compute final skewness
    variance = m2 / n
    std_dev = torch.sqrt(variance)
    m3_centered = m3 / n
    return m3_centered / (std_dev**3)  # Skewness


# CPU reference
def skewness_cpu(x):
    mean = x.mean()
    std = x.std(unbiased=False)
    m3 = ((x - mean) ** 3).mean()
    return m3 / (std**3)


def test_skewness():
    torch.manual_seed(42)
    x = torch.randn(100000, device="cuda")

    cpu_result = skewness_cpu(x.cpu())
    gpu_result = skewness_gpu(x)

    print(f"CPU: {cpu_result.item():.6f}")
    print(f"GPU: {gpu_result.item():.6f}")
    print(f"Difference: {abs(cpu_result - gpu_result).item():.2e}")

    assert torch.allclose(cpu_result, gpu_result, atol=1e-4), "Test failed"
    print("Test passed!")


def benchmark_skewness():
    sizes = [10**6, 10**7]
    for n in sizes:
        x = torch.randn(n, device="cuda")

        # CPU
        start = time.time()
        skewness_cpu(x.cpu())
        cpu_time = (time.time() - start) * 1000

        # GPU
        torch.cuda.synchronize()
        start = time.time()
        skewness_gpu(x)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) * 1000

        print(
            f"Size {n}: CPU {cpu_time:.1f}ms, GPU {gpu_time:.1f}ms, Speedup {cpu_time / gpu_time:.1f}x"
        )


test_skewness()
benchmark_skewness()
