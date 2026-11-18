# Lecture 5: how the error of C = A @ B grows with the inner dimension K (Exercise 2).
# Inputs are exactly representable in the input format, so the error comes only
# from accumulation and from rounding the output. Reference: FP64.
import torch
from kernels import matmul

torch.manual_seed(0)
M = N = 1024
one = torch.ones((), device="cuda")
rel = lambda C, R: ((C.double() - R).norm() / R.norm()).item()
print(f"{'K':>6s} {'inputs':>7s} | {'BF16 in, FP32 out':>17s} {'BF16 out':>9s} {'BF16 accumulator':>16s} | {'FP8 fast acc.':>13s} {'FP8 promoted':>12s}")
for K in (256, 1024, 4096, 16384, 65536):
    for kind in ("U(0,1)", "N(0,1)"):
        gen = torch.rand if kind == "U(0,1)" else torch.randn
        A, B = gen(M, K, device="cuda"), gen(K, N, device="cuda")
        a, b = A.bfloat16(), B.bfloat16()
        R = a.double() @ b.double()
        e1 = rel(torch.mm(a, b, out_dtype=torch.float32), R)      # cuBLAS, FP32 accumulate, FP32 out
        e2 = rel(a @ b, R)                                         # cuBLAS, BF16 out
        e3 = rel(matmul(a, b, acc="bf16"), R)                      # Triton, BF16 accumulator
        x = A.to(torch.float8_e4m3fn)
        y = B.t().contiguous().to(torch.float8_e4m3fn).t()
        R8 = x.double() @ y.double()
        f = lambda fast: torch._scaled_mm(x, y, scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=fast)
        print(f"{K:6d} {kind:>7s} | {e1:17.1e} {e2:9.1e} {e3:16.1e} | {rel(f(True), R8):13.1e} {rel(f(False), R8):12.1e}")
