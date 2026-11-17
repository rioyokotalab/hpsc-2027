import torch
import triton
import triton.language as tl
from bench import bench

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                 # which block of BLOCK elements
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N                        # the last block may be partial
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)

def add(a, b):
    c = torch.empty_like(a)
    N = c.numel()
    grid = (triton.cdiv(N, 1024),)
    add_kernel[grid](a, b, c, N, BLOCK=1024)
    return c

if __name__ == "__main__":
    for e in range(10, 29, 2):
        N = 1 << e
        a = torch.arange(N, device="cuda", dtype=torch.float32) % 1000
        b = torch.ones(N, device="cuda", dtype=torch.float32)
        assert torch.equal(add(a, b), a + b)
        t_triton = bench(lambda: add(a, b)) / 1e3
        t_torch = bench(lambda: a + b) / 1e3
        print(f"N 2^{e:<2}  Triton {12*N/t_triton/1e9:7.1f} GB/s  PyTorch {12*N/t_torch/1e9:7.1f} GB/s")
