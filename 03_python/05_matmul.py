import sys
import torch
import triton
import triton.language as tl
from bench import bench

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  PREC: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)   # rows of this C tile
    rn = tl.program_id(1) * BN + tl.arange(0, BN)   # columns of this C tile
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a_tile = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b_tile = tl.load(b, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a_tile, b_tile, input_precision=PREC)
        a += BK * sak
        b += BK * sbk
    c = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

def matmul(a, b, prec="ieee", BM=128, BN=128, BK=32, num_warps=4, num_stages=3):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    return matmul_kernel[grid](a, b, c, M, N, K,
                               a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, PREC=prec,
                               num_warps=num_warps, num_stages=num_stages), c

def check(c, a, b):
    ref = (a.double() @ b.double())
    scale = (a.double().abs() @ b.double().abs())
    return ((c.double() - ref).abs() / scale).max().item()

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    a = torch.randn(N, N, device="cuda")
    b = torch.randn(N, N, device="cuda")
    tflops = lambda ms: 2 * N**3 / (ms / 1e3) / 1e12
    for name, prec, tf32, cfg in (("FP32", "ieee", False, (128, 128, 32, 4, 3)),
                                  ("TF32", "tf32", True, (128, 128, 32, 8, 3))):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        _, c = matmul(a, b, prec, *cfg)
        err_t, err_c = check(c, a, b), check(a @ b, a, b)
        t_t = bench(lambda: matmul(a, b, prec, *cfg))
        t_c = bench(lambda: a @ b)
        print(f"{name}  Triton {tflops(t_t):6.1f} TFLOP/s (error {err_t:.1e})  "
              f"cuBLAS {tflops(t_c):6.1f} TFLOP/s (error {err_c:.1e})")
    a16, b16 = a.bfloat16(), b.bfloat16()
    cfg = (128, 256, 64, 8, 4)
    _, c = matmul(a16, b16, "ieee", *cfg)
    t_t = bench(lambda: matmul(a16, b16, "ieee", *cfg))
    t_c = bench(lambda: a16 @ b16)
    print(f"BF16  Triton {tflops(t_t):6.1f} TFLOP/s (error {check(c, a16, b16):.1e})  "
          f"cuBLAS {tflops(t_c):6.1f} TFLOP/s (error {check(a16 @ b16, a16, b16):.1e})")
