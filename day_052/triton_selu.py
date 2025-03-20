import torch
import triton
import triton.language as tl
import time


# Triton Kernel for SELU
@triton.jit
def selu_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
    LAMBDA: tl.constexpr,  # Scaling factor for SELU
    ALPHA: tl.constexpr,  # Negative slope for SELU
):
    pid = tl.program_id(axis=0)  # Index of the current program (thread block)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to avoid out-of-bounds access

    # Load data from input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute SELU: y = λ * (x if x > 0 else α * (e^x - 1))
    y = tl.where(x > 0, LAMBDA * x, LAMBDA * ALPHA * (tl.exp(x) - 1.0))
    # Store result to output tensor
    tl.store(y_ptr + offsets, y, mask=mask)


# Wrapper function to call the Triton kernel
def selu_triton(x: torch.Tensor):
    y = torch.empty_like(x)  # Output tensor
    n_elements = x.numel()  # Total number of elements
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # Grid function
    # Constants for SELU
    LAMBDA = 1.0507009873554804934193349852946
    ALPHA = 1.6732632423543772848170429916717
    selu_kernel[grid](
        x, y, n_elements, BLOCK_SIZE=1024, LAMBDA=LAMBDA, ALPHA=ALPHA
    )  # Launch kernel
    return y


# CPU Implementation of SELU for Comparison
def selu_cpu(x: torch.Tensor):
    LAMBDA = 1.0507009873554804934193349852946
    ALPHA = 1.6732632423543772848170429916717
    return torch.where(x > 0, LAMBDA * x, LAMBDA * ALPHA * (torch.exp(x) - 1.0))


# Test Function
def test_selu():
    # Create a random tensor
    size = 1 << 20  # 1 million elements
    x = torch.rand(size, device="cuda") * 2 - 1  # Values between -1 and 1

    # Compute results using Triton and CPU
    y_triton = selu_triton(x)
    y_cpu = selu_cpu(x.cpu()).cuda()  # Move to GPU for comparison

    # Validate correctness
    assert torch.allclose(y_triton, y_cpu, rtol=1e-4, atol=1e-6), (
        "Triton and CPU results do not match!"
    )
    print("Results match! Triton implementation is correct.")


# Performance Comparison
def performance_comparison():
    size = 1 << 20  # 1 million elements
    x = torch.rand(size, device="cuda") * 2 - 1  # Values between -1 and 1

    # Warm-up (to avoid initial overhead)
    for _ in range(10):
        selu_triton(x)

    # Benchmark Triton
    start_time = time.time()
    for _ in range(100):
        selu_triton(x)
    triton_time = time.time() - start_time
    print(f"Triton Execution Time: {triton_time:.6f} seconds")

    # Benchmark CPU
    x_cpu = x.cpu()
    start_time = time.time()
    for _ in range(100):
        selu_cpu(x_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU Execution Time: {cpu_time:.6f} seconds")


# Run tests and performance comparison
if __name__ == "__main__":
    test_selu()
    performance_comparison()
