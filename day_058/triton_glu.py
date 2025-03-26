import torch
import torch.nn as nn
import time


# CUDA Kernel for GLU
class GLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Split input into two halves
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        x1, x2 = x.chunk(2, dim=-1)
        sig_x2 = torch.sigmoid(x2)

        # Gradients for x1 and x2
        grad_x1 = grad_output * sig_x2
        grad_x2 = grad_output * x1 * sig_x2 * (1 - sig_x2)

        # Combine gradients
        return torch.cat([grad_x1, grad_x2], dim=-1)


def glu_cuda(x):
    return GLUFunction.apply(x)


# CPU Implementation for comparison
def glu_cpu(x):
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * torch.sigmoid(x2)


# Test Function
def test_glu():
    # Create random input
    batch_size, seq_len, hidden_size = 32, 64, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", requires_grad=True)

    # Forward pass
    y_cuda = glu_cuda(x)
    y_cpu = glu_cpu(x.cpu()).cuda()

    # Backward pass
    grad = torch.randn_like(y_cuda)
    y_cuda.backward(grad)
    grad_cuda = x.grad

    x.grad = None
    x_cpu = x.detach().cpu().requires_grad_(True)
    glu_cpu(x_cpu).backward(grad.cpu())
    grad_cpu = x_cpu.grad.cuda()

    # Validate
    forward_diff = (y_cuda - y_cpu).abs().max().item()
    backward_diff = (grad_cuda - grad_cpu).abs().max().item()

    print(f"Max forward difference: {forward_diff:.2e}")
    print(f"Max backward difference: {backward_diff:.2e}")

    assert torch.allclose(y_cuda, y_cpu, rtol=1e-5), "Forward pass mismatch"
    assert torch.allclose(grad_cuda, grad_cpu, rtol=1e-5), "Backward pass mismatch"
    print("Test passed!")


# Performance Comparison
def performance_comparison():
    batch_size, seq_len, hidden_size = 32, 64, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

    # Warm-up
    for _ in range(10):
        _ = glu_cuda(x)

    # Benchmark CUDA
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = glu_cuda(x)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    # Benchmark CPU
    x_cpu = x.cpu()
    start = time.time()
    for _ in range(1000):
        _ = glu_cpu(x_cpu)
    cpu_time = time.time() - start

    print(f"CUDA Time: {cuda_time:.4f} sec")
    print(f"CPU Time: {cpu_time:.4f} sec")
    print(f"Speedup: {cpu_time / cuda_time:.2f}x")


if __name__ == "__main__":
    test_glu()
    performance_comparison()
