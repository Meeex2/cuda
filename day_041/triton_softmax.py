import torch
import triton
import triton.language as tl
import time


@triton.jit
def max_kernel(
    input_ptr,
    max_output_ptr,
    input_row_stride,
    max_output_row_stride,
    n_cols,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    col_start = block_idx * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets

    row_chunk = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    block_max = tl.max(row_chunk, axis=0)

    max_output_ptr = max_output_ptr + row_idx * max_output_row_stride + block_idx
    tl.store(max_output_ptr, block_max, mask=(col_start < n_cols))


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    row_max_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    row_max = tl.load(row_max_ptr + row_idx)

    col_start = block_idx * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets

    row_chunk = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row_chunk - row_max
    exp_row = tl.exp(row_minus_max)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, exp_row, mask=col_offsets < n_cols)


def softmax_triton(input: torch.Tensor) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on the GPU"

    n_rows, n_cols = input.shape
    output = torch.empty_like(input)

    BLOCK_SIZE = 1024
    num_blocks_per_row = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (n_rows, num_blocks_per_row)

    max_per_block = torch.empty(
        (n_rows, num_blocks_per_row), dtype=torch.float32, device="cuda"
    )
    max_kernel[grid](
        input,
        max_per_block,
        input.stride(0),
        max_per_block.stride(0),
        n_cols,
        num_blocks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    row_max = torch.max(max_per_block, dim=1)[0]

    softmax_kernel[grid](
        input,
        output,
        row_max,
        input.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    row_sums = torch.sum(output, dim=1, keepdim=True)
    output /= row_sums

    return output


def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    torch.cuda.synchronize()
    return result, time.time() - start


if __name__ == "__main__":
    n_rows, n_cols = 4096, 4096
    input = torch.rand(n_rows, n_cols, device="cuda")

    output_triton, triton_time = measure_time(softmax_triton, input)
    print(f"Triton kernel time: {triton_time * 1000:.4f} ms")

    output_torch_gpu, torch_gpu_time = measure_time(lambda: torch.softmax(input, dim=1))
    print(f"PyTorch (GPU) Softmax time: {torch_gpu_time * 1000:.4f} ms")

    assert torch.allclose(output_triton, output_torch_gpu, atol=1e-3, rtol=1e-3), (
        "Validation failed!"
    )
    print("Validation passed!")
