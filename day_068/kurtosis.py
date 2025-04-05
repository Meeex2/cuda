import torch
import triton
import triton.language as tl

# First kernel: Compute mean
@triton.jit
def mean_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    block_sum = tl.sum(x)
    tl.atomic_add(output_ptr, block_sum)

# Second kernel: Compute centered moments
@triton.jit
def moments_kernel(
    x_ptr,
    mean,
    n_elements,
    m2_ptr,
    m4_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    x_centered = x - tl.load(mean)
    x2 = x_centered * x_centered
    x4 = x2 * x2
    
    tl.atomic_add(m2_ptr, tl.sum(x2))
    tl.atomic_add(m4_ptr, tl.sum(x4))

def kurtosis_gpu(x):
    n = x.numel()
    
    # Compute mean
    mean = torch.zeros(1, device=x.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    mean_kernel[grid](x, mean, n, BLOCK_SIZE=BLOCK_SIZE)
    mean = mean / n
    
    # Compute centered moments
    m2 = torch.zeros(1, device=x.device)
    m4 = torch.zeros(1, device=x.device)
    moments_kernel[grid](x, mean, n, m2, m4, BLOCK_SIZE=BLOCK_SIZE)
    
    variance = m2 / n
    m4_centered = m4 / n
    return (m4_centered / (variance ** 2)) - 3  # Excess kurtosis

# CPU reference remains the same
def kurtosis_cpu(x):
    mean = x.mean()
    variance = x.var(unbiased=False)
    m4 = ((x - mean)**4).mean()
    return (m4 / variance**2) - 3

# Test and benchmark functions
def test_kurtosis():
    torch.manual_seed(42)
    x = torch.randn(100000, device='cuda')
    
    cpu_result = kurtosis_cpu(x.cpu())
    gpu_result = kurtosis_gpu(x)
    
    print(f"CPU: {cpu_result.item():.4f}")
    print(f"GPU: {gpu_result.item():.4f}")
    print(f"Difference: {abs(cpu_result - gpu_result).item():.2e}")
    
    assert torch.allclose(cpu_result, gpu_result, atol=1e-4), "Test failed"
    print("Test passed!")

def benchmark():
    sizes = [10**6, 10**7, 10**8]
    for n in sizes:
        x = torch.randn(n, device='cuda')
        
        # CPU
        start = time.time()
        kurtosis_cpu(x.cpu())
        cpu_time = (time.time() - start) * 1000
        
        # GPU
        torch.cuda.synchronize()
        start = time.time()
        kurtosis_gpu(x)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) * 1000
        
        print(f"Size {n}: CPU {cpu_time:.1f}ms, GPU {gpu_time:.1f}ms, Speedup {cpu_time/gpu_time:.1f}x")

test_kurtosis()
benchmark()