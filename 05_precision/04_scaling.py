# Lecture 5: scaling factors and outliers (Exercise 3).
# X has 8 outlier columns multiplied by a factor; every other entry is N(0,1).
# We quantize X with one scale per tensor or per block and measure what happens
# to the ordinary entries.
import torch
from bench import bench

torch.manual_seed(0)
M = K = 4096
X = torch.randn(M, K, device="cuda")
cols = torch.randperm(K, device="cuda")[:8]
normal = torch.ones(K, dtype=torch.bool, device="cuda")
normal[cols] = False
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")   # FP4 magnitudes

def cast(y, fmt):
    if fmt == "INT8":
        return y.round()
    if fmt == "FP8":
        return y.to(torch.float8_e4m3fn).float()
    i = (y.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)             # FP4: nearest E2M1 value
    return torch.sign(y) * E2M1[i]

def quantize(x, fmt, block):
    top = {"INT8": 127, "FP8": 448, "FP4": 6}[fmt]                  # largest value of the format
    xb = x.reshape(1, -1) if block == 0 else x.reshape(M, K // block, block)
    s = xb.abs().amax(-1, keepdim=True) / top                       # one scale per tensor or block
    return (cast(xb / s, fmt) * s).reshape(M, K)

schemes = [("INT8", 0), ("FP8", 0), ("FP4", 0), ("FP8", 32), ("FP4", 32), ("FP4", 16)]
print("Relative error of the ordinary entries (fraction flushed to zero)")
print(f"{'outlier':>8s} " + " ".join(f"{f + (' tensor' if b == 0 else f' block{b}'):>17s}" for f, b in schemes))
for factor in (1, 10, 100, 1e3, 1e4, 1e5, 1e6):
    Xo = X.clone()
    Xo[:, cols] *= factor
    row = []
    for fmt, block in schemes:
        Xq = quantize(Xo, fmt, block)
        err = ((Xq - Xo)[:, normal].norm() / Xo[:, normal].norm()).item()
        zero = (Xq[:, normal] == 0).float().mean().item()
        row.append(f"{err:9.1e} ({zero:4.2f})")
    print(f"{factor:8.0e} " + " ".join(row))

print("\nFP8 GEMM, 8192^3: speed by scaling granularity")
N = 8192
A = torch.randn(N, N, device="cuda")
B = torch.randn(N, N, device="cuda")
ta, tb = A.abs().max() / 448, B.abs().max() / 448
x, y = (A / ta).to(torch.float8_e4m3fn), (B / tb).t().contiguous().to(torch.float8_e4m3fn).t()
t = bench(lambda: torch._scaled_mm(x, y, scale_a=ta, scale_b=tb, out_dtype=torch.bfloat16))
print(f"  per tensor   {2 * N**3 / t / 1e9:7.1f} TFLOP/s  (default: partial sums promoted to FP32)")
t = bench(lambda: torch._scaled_mm(x, y, scale_a=ta, scale_b=tb, out_dtype=torch.bfloat16, use_fast_accum=True))
print(f"  per tensor   {2 * N**3 / t / 1e9:7.1f} TFLOP/s  (fast accumulation)")
ra, rb = A.abs().amax(1, keepdim=True) / 448, B.abs().amax(0, keepdim=True) / 448
x, y = (A / ra).to(torch.float8_e4m3fn), (B / rb).t().contiguous().to(torch.float8_e4m3fn).t()
t = bench(lambda: torch._scaled_mm(x, y, scale_a=ra, scale_b=rb, out_dtype=torch.bfloat16))
print(f"  per row/col  {2 * N**3 / t / 1e9:7.1f} TFLOP/s")
a, b = A.bfloat16(), B.bfloat16()
t = bench(lambda: a @ b)
print(f"  BF16         {2 * N**3 / t / 1e9:7.1f} TFLOP/s")
t = bench(lambda: (A / A.abs().amax(1, keepdim=True) * 448).to(torch.float8_e4m3fn))
print(f"  quantizing A (FP32 -> FP8, per row) takes {t:.2f} ms")
