import torch
import triton
import triton.language as tl
import time


# Triton kernel for matrix-vector multiplication
@triton.jit
def matvec_kernel(
    matrix_ptr,  # Pointer to the matrix (2D tensor)
    vector_ptr,  # Pointer to the vector (1D tensor)
    output_ptr,  # Pointer to the output vector (1D tensor)
    n_rows,  # Number of rows in the matrix
    n_cols,  # Number of columns in the matrix
    BLOCK_SIZE: tl.constexpr,  # Number of rows each program should process
):
    row_idx = tl.program_id(axis=0)  # 1D launch grid (one thread block per row)
    row_start = row_idx * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds accesses
    row_mask = row_offsets < n_rows

    # Initialize the output value for this row
    output_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over columns to compute the dot product
    for col_idx in range(0, n_cols):
        # Load matrix and vector elements
        matrix_elements = tl.load(
            matrix_ptr + row_offsets * n_cols + col_idx, mask=row_mask
        )
        vector_element = tl.load(vector_ptr + col_idx)

        # Accumulate the dot product
        output_value += matrix_elements * vector_element

    # Store the result back to global memory
    tl.store(output_ptr + row_offsets, output_value, mask=row_mask)


# Triton matrix-vector multiplication wrapper
def matvec_triton(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    assert matrix.is_cuda and vector.is_cuda
    assert matrix.shape[1] == vector.shape[0], "Matrix columns must match vector length"

    n_rows, n_cols = matrix.shape
    output = torch.empty(n_rows, device="cuda")

    # Configure the kernel
    BLOCK_SIZE = 128  # Number of rows processed per thread block
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)

    # Launch kernel
    matvec_kernel[grid](matrix, vector, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)

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
    n_rows, n_cols = 4096, 4096  # 4096x4096 matrix and 4096-element vector
    matrix = torch.rand(n_rows, n_cols, device="cuda")
    vector = torch.rand(n_cols, device="cuda")

    # Measure Triton kernel time
    output_triton, triton_time = measure_time(matvec_triton, matrix, vector)
    print(f"Triton kernel time: {triton_time * 1000:.4f} ms")

    # Measure PyTorch matrix-vector multiplication time
    output_torch, torch_time = measure_time(lambda: matrix @ vector)
    print(f"PyTorch matrix-vector multiplication time: {torch_time * 1000:.4f} ms")

    # Validate results
    assert torch.allclose(output_triton, output_torch, atol=1e-5)
    print("Validation passed!")
