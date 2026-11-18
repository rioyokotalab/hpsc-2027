# Lecture 5: one matrix multiplication in six precisions: speed and error against FP64.
import sys, torch
from bench import bench

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
torch.manual_seed(0)
A = torch.randn(N, N, device="cuda", dtype=torch.float64)
B = torch.randn(N, N, device="cuda", dtype=torch.float64)
R = A @ B                                         # FP64 reference

def report(name, f):
    t = bench(f, reps=5)
    err = ((f().double() - R).norm() / R.norm()).item()
    print(f"N={N} {name:28s} {2 * N**3 / t / 1e9:8.1f} TFLOP/s   relative error {err:.2e}")

report("FP64", lambda: A @ B)
a, b = A.float(), B.float()
torch.backends.cuda.matmul.allow_tf32 = False     # PyTorch's default for matmul
report("FP32", lambda: a @ b)
torch.backends.cuda.matmul.allow_tf32 = True
report("TF32 (FP32 in, tensor cores)", lambda: a @ b)
torch.backends.cuda.matmul.allow_tf32 = False
for dt, name in ((torch.bfloat16, "BF16"), (torch.float16, "FP16")):
    x, y = A.to(dt), B.to(dt)
    report(name, lambda: x @ y)
# FP8 E4M3 with one scale per tensor; cuBLAS wants B column-major
sa = (A.abs().max() / 448).float()
sb = (B.abs().max() / 448).float()
x = (A / sa).to(torch.float8_e4m3fn)
y = (B / sb).t().contiguous().to(torch.float8_e4m3fn).t()
report("FP8 E4M3, per-tensor scale", lambda: torch._scaled_mm(x, y, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
