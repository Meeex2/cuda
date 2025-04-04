import torch
import triton
import triton.language as tl


@triton.jit
def fisher_kernel(
    log_probs_ptr,  # Pointer to input gradients [n_samples, n_params]
    fisher_ptr,  # Pointer to output matrix [n_params, n_params]
    n_samples,  # Number of samples
    n_params,  # Number of parameters
    stride_sample,  # Stride between samples (log_probs.stride(0))
    stride_param,  # Stride between parameters (log_probs.stride(1))
    BLOCK_SIZE: tl.constexpr,
):
    # 2D launch grid for symmetric matrix
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Block offsets
    p0_offsets = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    p1_offsets = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create masks for valid indices
    p0_mask = p0_offsets < n_params
    p1_mask = p1_offsets < n_params
    block_mask = p0_mask[:, None] & p1_mask[None, :]  # 2D block mask

    # Initialize accumulator
    block_fisher = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Accumulate outer products
    for i in range(n_samples):
        # Load current sample's gradients
        sample_ptr = log_probs_ptr + i * stride_sample
        grad_p0 = tl.load(sample_ptr + p0_offsets * stride_param, mask=p0_mask)
        grad_p1 = tl.load(sample_ptr + p1_offsets * stride_param, mask=p1_mask)

        # Compute outer product and accumulate
        outer = grad_p0[:, None] * grad_p1[None, :]
        block_fisher += tl.where(block_mask, outer, 0.0)

    # Normalize and store
    block_fisher /= n_samples
    offsets = p0_offsets[:, None] * n_params + p1_offsets[None, :]
    tl.store(fisher_ptr + offsets, block_fisher, mask=block_mask)


def fisher_information_gpu(log_probs):
    n_samples, n_params = log_probs.shape
    fisher = torch.zeros((n_params, n_params), device=log_probs.device)

    # Tune BLOCK_SIZE based on GPU capabilities (must be power of 2)
    BLOCK_SIZE = 32 if triton.next_power_of_2(n_params) >= 32 else 16

    grid = (
        triton.cdiv(n_params, BLOCK_SIZE),
        triton.cdiv(n_params, BLOCK_SIZE),
    )

    fisher_kernel[grid](
        log_probs,
        fisher,
        n_samples,
        n_params,
        log_probs.stride(0),
        log_probs.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return fisher


# Testing and Benchmarking
def test_fisher_information():
    torch.manual_seed(42)
    n_samples, n_params = 1000, 128
    log_probs = torch.randn(n_samples, n_params, device="cuda")

    # Compute results
    cpu_result = fisher_information_cpu(log_probs.cpu()).cuda()
    gpu_result = fisher_information_gpu(log_probs)

    # Check symmetry
    assert torch.allclose(gpu_result, gpu_result.T, atol=1e-6), (
        "Fisher matrix not symmetric"
    )

    # Check values
    max_diff = torch.max(torch.abs(cpu_result - gpu_result)).item()
    print(f"Max difference between CPU and GPU: {max_diff:.2e}")
    assert max_diff < 1e-5, "CPU and GPU results differ too much"

    print("All tests passed!")


def benchmark_fisher():
    sizes = [(10**3, 64), (10**4, 64), (10**3, 128), (10**4, 128)]
    cpu_times = []
    gpu_times = []

    for n_samples, n_params in sizes:
        print(f"\nBenchmarking {n_samples} samples, {n_params} parameters")
        log_probs = torch.randn(n_samples, n_params, device="cuda")

        # CPU benchmark
        log_probs_cpu = log_probs.cpu()
        start = time.time()
        fisher_information_cpu(log_probs_cpu)
        cpu_time = (time.time() - start) * 1000
        cpu_times.append(cpu_time)

        # GPU benchmark
        torch.cuda.synchronize()
        start = time.time()
        fisher_information_gpu(log_probs)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) * 1000
        gpu_times.append(gpu_time)

        print(
            f"CPU: {cpu_time:.2f}ms, GPU: {gpu_time:.2f}ms, Speedup: {cpu_time / gpu_time:.1f}x"
        )

    # Plot results
    labels = [f"{s}x{p}" for s, p in sizes]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(sizes)) - 0.2, cpu_times, width=0.4, label="CPU")
    plt.bar(np.arange(len(sizes)) + 0.2, gpu_times, width=0.4, label="GPU")
    plt.xticks(np.arange(len(sizes)), labels)
    plt.xlabel("Problem size (samples × parameters)")
    plt.ylabel("Time (ms)")
    plt.title("Fisher Information Computation Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("fisher_performance.png")
    plt.show()


if __name__ == "__main__":
    test_fisher_information()
    benchmark_fisher()
