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
