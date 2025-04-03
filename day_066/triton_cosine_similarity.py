import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time
import numpy as np

# Cosine similarity CPU implementation
def cosine_similarity_cpu(x, y):
    x_norm = torch.norm(x, dim=1)
    y_norm = torch.norm(y, dim=1)
    dot_product = (x * y).sum(dim=1)
    return dot_product / (x_norm * y_norm + 1e-8)

@triton.jit
def cosine_similarity_kernel(
    x_ptr,          # Pointer to first input tensor
    y_ptr,          # Pointer to second input tensor
    output_ptr,     # Pointer to output tensor
    n_vectors,      # Number of vectors
    vector_size,    # Size of each vector
    BLOCK_SIZE: tl.constexpr,      # Vectors per block
    VECTOR_BLOCK: tl.constexpr,    # Elements per vector to process
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_vectors
    
    # Initialize accumulators
    dot = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    x_norm = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    y_norm = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process vector elements in blocks
    for i in range(0, vector_size, VECTOR_BLOCK):
        vec_offsets = i + tl.arange(0, VECTOR_BLOCK)
        vec_mask = vec_offsets < vector_size
        
        # Load current blocks of vectors
        x = tl.load(x_ptr + offsets[:, None] * vector_size + vec_offsets[None, :], 
                   mask=mask[:, None] & vec_mask[None, :])
        y = tl.load(y_ptr + offsets[:, None] * vector_size + vec_offsets[None, :], 
                   mask=mask[:, None] & vec_mask[None, :])
        
        # Accumulate dot product and norms
        dot += tl.sum(x * y, axis=1)
        x_norm += tl.sum(x * x, axis=1)
        y_norm += tl.sum(y * y, axis=1)
    
    # Compute cosine similarity
    x_norm = tl.sqrt(x_norm)
    y_norm = tl.sqrt(y_norm)
    similarity = dot / (x_norm * y_norm + 1e-8)
    
    # Store results
    tl.store(output_ptr + offsets, similarity, mask=mask)

def cosine_similarity_gpu(x, y):
    assert x.shape == y.shape
    assert x.dim() == 2, "Inputs must be 2D tensors"
    
    n_vectors, vector_size = x.shape
    output = torch.empty(n_vectors, device=x.device, dtype=x.dtype)
    
    # Configuration
    BLOCK_SIZE = 128  # Vectors per block
    VECTOR_BLOCK = 64  # Elements per vector to process at once
    
    grid = lambda meta: (triton.cdiv(n_vectors, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    cosine_similarity_kernel[grid](
        x, y, output,
        n_vectors, vector_size,
        BLOCK_SIZE=BLOCK_SIZE,
        VECTOR_BLOCK=VECTOR_BLOCK
    )
    
    return output

# Benchmarking function
def benchmark(fn, x, y, num_warmups=10, num_iters=100):
    # Warmup
    for _ in range(num_warmups):
        fn(x, y)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        fn(x, y)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / num_iters
    return elapsed_ms

# Test function with detailed error reporting
def test_cosine_similarity():
    # Test data
    torch.manual_seed(42)
    n_vectors = 1000
    vector_size = 256
    
    x = torch.randn(n_vectors, vector_size, device='cuda')
    y = torch.randn(n_vectors, vector_size, device='cuda')
    
    # Compute results
    cpu_result = cosine_similarity_cpu(x.cpu(), y.cpu()).cuda()
    gpu_result = cosine_similarity_gpu(x, y)
    pytorch_result = torch.nn.functional.cosine_similarity(x, y, dim=1)
    
    # Print values for debugging
    print(f"CPU result (first 5): {cpu_result[:5]}")
    print(f"Triton GPU result (first 5): {gpu_result[:5]}")
    print(f"PyTorch GPU result (first 5): {pytorch_result[:5]}")
    
    # Calculate absolute differences
    diff_triton = torch.max(torch.abs(cpu_result - gpu_result)).item()
    diff_pytorch = torch.max(torch.abs(cpu_result - pytorch_result)).item()
    print(f"Max difference between CPU and Triton: {diff_triton}")
    print(f"Max difference between CPU and PyTorch: {diff_pytorch}")
    
    # Check correctness with tolerance
    atol = 1e-5
    assert torch.allclose(cpu_result, gpu_result, atol=atol), \
        f"CPU and Triton GPU implementations differ by {diff_triton} (beyond tolerance {atol})"
    assert torch.allclose(cpu_result, pytorch_result, atol=atol), \
        f"CPU and PyTorch GPU implementations differ by {diff_pytorch} (beyond tolerance {atol})"
    print("Test passed! All implementations match within tolerance.")

# Performance comparison
def compare_performance():
    sizes = [(10**i, 256) for i in range(2, 6)]  # (100 to 10000 vectors) × 256 dim
    cpu_times = []
    pytorch_times = []
    triton_times = []
    
    for n_vectors, vector_size in sizes:
        print(f"\nBenchmarking size: {n_vectors} vectors of dim {vector_size}")
        x = torch.randn(n_vectors, vector_size, device='cuda')
        y = torch.randn(n_vectors, vector_size, device='cuda')
        
        # Verify correctness first
        cpu_val = cosine_similarity_cpu(x.cpu(), y.cpu()).cuda()
        triton_val = cosine_similarity_gpu(x, y)
        if not torch.allclose(cpu_val, triton_val, atol=1e-5):
            print(f"Warning: Potential numerical issue at size {(n_vectors, vector_size)}")
            print(f"CPU: {cpu_val[:5]}, Triton: {triton_val[:5]}")
        
        # CPU benchmark
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        for _ in range(10):
            cosine_similarity_cpu(x_cpu, y_cpu)
        cpu_time = (time.time() - start) * 1000 / 10
        cpu_times.append(cpu_time)
        
        # PyTorch GPU benchmark
        pytorch_time = benchmark(torch.nn.functional.cosine_similarity, x, y, 1)
        pytorch_times.append(pytorch_time)
        
        # Triton benchmark
        triton_time = benchmark(cosine_similarity_gpu, x, y)
        triton_times.append(triton_time)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    x_labels = [f"{n}v" for n, d in sizes]
    plt.plot(x_labels, cpu_times, label='CPU')
    plt.plot(x_labels, pytorch_times, label='GPU (PyTorch)')
    plt.plot(x_labels, triton_times, label='GPU (Triton)')
    plt.xlabel('Number of vectors (each 256-dim)')
    plt.ylabel('Time (ms)')
    plt.title('Cosine Similarity Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('cosine_similarity_performance.png')
    plt.show()
    
    # Print results
    print("\nPerformance Results:")
    print(f"{'Size':>15} {'CPU (ms)':>10} {'PyTorch GPU (ms)':>15} {'Triton GPU (ms)':>15} {'Speedup (Triton vs CPU)':>20}")
    for i, (n_vectors, vector_size) in enumerate(sizes):
        speedup = cpu_times[i] / triton_times[i]
        size_str = f"{n_vectors}x{vector_size}"
        print(f"{size_str:15} {cpu_times[i]:10.3f} {pytorch_times[i]:15.3f} {triton_times[i]:15.3f} {speedup:20.1f}x")

if __name__ == "__main__":
    test_cosine_similarity()
    compare_performance()