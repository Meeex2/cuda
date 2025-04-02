import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time
import numpy as np


# Triplet Loss CPU implementation matching PyTorch's formula
def triplet_loss_cpu(anchor, positive, negative, margin=1.0, reduction="mean"):
    distance_positive = (anchor - positive).pow(2).sum(1).sqrt()  # L2 distance
    distance_negative = (anchor - negative).pow(2).sum(1).sqrt()  # L2 distance
    losses = torch.relu(distance_positive - distance_negative + margin)

    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    return losses


@triton.jit
def triplet_loss_kernel(
    anchor_ptr,  # Pointer to anchor tensor
    positive_ptr,  # Pointer to positive tensor
    negative_ptr,  # Pointer to negative tensor
    output_ptr,  # Pointer to output tensor
    margin,  # Margin value
    n_vectors,  # Number of triplets
    vector_size,  # Size of each embedding vector
    BLOCK_SIZE: tl.constexpr,  # Number of vectors each program should process
    VECTOR_BLOCK: tl.constexpr,  # Elements per vector to process at once
):
    pid = tl.program_id(axis=0)  # We use 1D launch grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_vectors

    # Initialize accumulators for distances
    pos_dist = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    neg_dist = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Process each vector in blocks for better memory efficiency
    for i in range(0, vector_size, VECTOR_BLOCK):
        vec_offsets = i + tl.arange(0, VECTOR_BLOCK)
        vec_mask = vec_offsets < vector_size

        # Load current blocks of vectors
        a = tl.load(
            anchor_ptr + offsets[:, None] * vector_size + vec_offsets[None, :],
            mask=mask[:, None] & vec_mask[None, :],
        )
        p = tl.load(
            positive_ptr + offsets[:, None] * vector_size + vec_offsets[None, :],
            mask=mask[:, None] & vec_mask[None, :],
        )
        n = tl.load(
            negative_ptr + offsets[:, None] * vector_size + vec_offsets[None, :],
            mask=mask[:, None] & vec_mask[None, :],
        )

        # Accumulate squared differences
        pos_dist += tl.sum((a - p) * (a - p), axis=1)
        neg_dist += tl.sum((a - n) * (a - n), axis=1)

    # Compute L2 distances and then triplet loss
    pos_dist = tl.sqrt(pos_dist)
    neg_dist = tl.sqrt(neg_dist)
    loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)

    # Store results
    tl.store(output_ptr + offsets, loss, mask=mask)


def triplet_loss_gpu(anchor, positive, negative, margin=1.0, reduction="mean"):
    assert anchor.shape == positive.shape == negative.shape
    assert anchor.dim() == 2, "Inputs must be 2D tensors"

    n_vectors, vector_size = anchor.shape
    output = torch.empty(n_vectors, device=anchor.device, dtype=anchor.dtype)

    # Configuration
    BLOCK_SIZE = 128  # Number of triplets to process per block
    VECTOR_BLOCK = 64  # Elements of each vector to process at once

    grid = lambda meta: (triton.cdiv(n_vectors, meta["BLOCK_SIZE"]),)

    # Launch kernel
    triplet_loss_kernel[grid](
        anchor,
        positive,
        negative,
        output,
        margin,
        n_vectors,
        vector_size,
        BLOCK_SIZE=BLOCK_SIZE,
        VECTOR_BLOCK=VECTOR_BLOCK,
    )

    if reduction == "mean":
        return output.mean()
    elif reduction == "sum":
        return output.sum()
    return output


def test_triplet_loss():
    # Test data
    torch.manual_seed(42)  # For reproducibility
    n_vectors = 1000
    vector_size = 256
    margin = 1.0

    anchor = torch.randn(n_vectors, vector_size, device="cuda")
    positive = torch.randn(n_vectors, vector_size, device="cuda")
    negative = torch.randn(n_vectors, vector_size, device="cuda")

    # Compute results
    cpu_result = triplet_loss_cpu(
        anchor.cpu(), positive.cpu(), negative.cpu(), margin
    ).cuda()
    gpu_result = triplet_loss_gpu(anchor, positive, negative, margin)
    pytorch_result = torch.nn.functional.triplet_margin_loss(
        anchor, positive, negative, margin=margin, reduction="mean", p=2, eps=1e-6
    )

    # Print values for debugging
    print(f"CPU result: {cpu_result.item()}")
    print(f"Triton GPU result: {gpu_result.item()}")
    print(f"PyTorch GPU result: {pytorch_result.item()}")

    # Calculate absolute differences
    diff_triton = torch.abs(cpu_result - gpu_result).item()
    diff_pytorch = torch.abs(cpu_result - pytorch_result).item()
    print(f"Difference between CPU and Triton: {diff_triton}")
    print(f"Difference between CPU and PyTorch: {diff_pytorch}")

    # Check correctness with tolerance
    atol = 1e-5
    assert torch.allclose(cpu_result, gpu_result, atol=atol), (
        f"CPU and Triton GPU implementations differ by {diff_triton} (beyond tolerance {atol})"
    )
    assert torch.allclose(cpu_result, pytorch_result, atol=atol), (
        f"CPU and PyTorch GPU implementations differ by {diff_pytorch} (beyond tolerance {atol})"
    )
    print("Test passed! All implementations match within tolerance.")


if __name__ == "__main__":
    test_triplet_loss()
