import torch
import triton
import triton.language as tl
import time


# Triton kernel for vector addition
@triton.jit
def vector_add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    pid = tl.program_id(axis=0)  # 1D launch grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds accesses
    mask = offsets < n_elements

    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute vector addition
    output = x + y

    # Store result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)


# Triton vector addition wrapper
def vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n_elements = output.numel()

    # Configure the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


# Timing function
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


# Test the Triton kernel
if __name__ == "__main__":
    # Generate test data
    size = 1000000  # 1 million elements
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    # Measure Triton kernel time
    output_triton, triton_time = measure_time(vector_add_triton, x, y)
    print(f"Triton kernel time: {triton_time * 1000:.4f} ms")

    # Measure PyTorch vector addition time
    output_torch, torch_time = measure_time(lambda: x + y)
    print(f"PyTorch vector addition time: {torch_time * 1000:.4f} ms")

    # Validate results
    assert torch.allclose(output_triton, output_torch, atol=1e-6)
    print("Validation passed!")
