import torch
import triton
import triton.language as tl
import time


# Triton Kernel for KL Divergence
@triton.jit
def kl_divergence_kernel(
    p_ptr,  # Pointer to the first probability distribution (P)
    q_ptr,  # Pointer to the second probability distribution (Q)
    output_ptr,  # Pointer to the output (scalar result)
    n_elements,  # Number of elements in the distributions
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Index of the current program (thread block)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to avoid out-of-bounds access

    # Load data from input tensors
    p = tl.load(p_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)
    # Compute KL Divergence: p * log(p / q)
    kl = p * tl.log(p / q)
    # Store partial results
    tl.store(output_ptr + offsets, kl, mask=mask)


# Wrapper function to call the Triton kernel
def kl_divergence_triton(p: torch.Tensor, q: torch.Tensor):
    n_elements = p.numel()  # Total number of elements
    output = torch.zeros_like(p)  # Output tensor for partial results
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # Grid function
    kl_divergence_kernel[grid](
        p, q, output, n_elements, BLOCK_SIZE=1024
    )  # Launch kernel
    # Sum the partial results to get the final KL Divergence
    return output.sum()


# CPU Implementation of KL Divergence for Comparison
def kl_divergence_cpu(p: torch.Tensor, q: torch.Tensor):
    return (p * (p / q).log()).sum()


# Test Function
def test_kl_divergence():
    # Create two random probability distributions
    size = 1 << 20  # 1 million elements
    p = torch.rand(size, device="cuda")
    q = torch.rand(size, device="cuda")
    # Normalize to ensure they are valid probability distributions
    p = p / p.sum()
    q = q / q.sum()

    # Compute results using Triton and CPU
    kl_triton = kl_divergence_triton(p, q)
    kl_cpu = kl_divergence_cpu(p.cpu(), q.cpu())

    # Validate correctness
    assert torch.allclose(kl_triton.cpu(), kl_cpu, rtol=1e-5), (
        "Triton and CPU results do not match!"
    )
    print("Results match! Triton implementation is correct.")


# Performance Comparison
def performance_comparison():
    # Create two random probability distributions
    size = 1 << 20  # 1 million elements
    p = torch.rand(size, device="cuda")
    q = torch.rand(size, device="cuda")
    # Normalize to ensure they are valid probability distributions
    p = p / p.sum()
    q = q / q.sum()

    # Warm-up (to avoid initial overhead)
    for _ in range(10):
        kl_divergence_triton(p, q)

    # Benchmark Triton
    start_time = time.time()
    for _ in range(100):
        kl_divergence_triton(p, q)
    triton_time = time.time() - start_time
    print(f"Triton Execution Time: {triton_time:.6f} seconds")

    # Benchmark CPU
    p_cpu = p.cpu()
    q_cpu = q.cpu()
    start_time = time.time()
    for _ in range(100):
        kl_divergence_cpu(p_cpu, q_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU Execution Time: {cpu_time:.6f} seconds")


# Run tests and performance comparison
if __name__ == "__main__":
    test_kl_divergence()
    performance_comparison()
