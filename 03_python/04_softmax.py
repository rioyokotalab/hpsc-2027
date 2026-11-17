import torch
import triton
import triton.language as tl
from bench import bench

def softmax_naive(x):                      # five kernels, each reads and writes memory
    m = x.max(dim=1, keepdim=True).values
    z = x - m
    e = torch.exp(z)
    s = e.sum(dim=1, keepdim=True)
    return e / s

@triton.jit
def softmax_kernel(x_ptr, y_ptr, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)                 # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)              # the row stays in registers
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride + cols, y, mask=mask)

def softmax_triton(x):
    M, N = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(M,)](x, y, N, x.stride(0), BLOCK=triton.next_power_of_2(N))
    return y

M = 8192
for N in (512, 1024, 2048, 4096, 8192):
    x = torch.randn(M, N, device="cuda")
    ref = torch.softmax(x, dim=1)
    assert torch.allclose(softmax_triton(x), ref, atol=1e-6)
    assert torch.allclose(softmax_naive(x), ref, atol=1e-6)
    gbs = lambda f: 2 * 4 * M * N / (bench(lambda: f(x)) / 1e3) / 1e9
    print(f"N {N:5}  naive {gbs(softmax_naive):7.1f} GB/s  "
          f"Triton {gbs(softmax_triton):7.1f} GB/s  torch.softmax "
          f"{gbs(lambda x: torch.softmax(x, dim=1)):7.1f} GB/s")
