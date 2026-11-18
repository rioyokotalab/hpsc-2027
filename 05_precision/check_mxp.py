# Lecture 5 homework checker: accuracy (HPL's scaled residual), time, and the kernels that ran.
import importlib, sys, torch
from bench import bench
mxp = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "10_mxp")

def problem(N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.rand(N, N, device="cuda", dtype=torch.float64, generator=g) - 0.5
    A.diagonal().add_(A.abs().sum(1))                         # diagonally dominant
    b = torch.rand(N, 1, device="cuda", dtype=torch.float64, generator=g) - 0.5
    return A, b

def scaled_residual(A, x, b):
    N = A.shape[0]
    r = (b - A @ x).abs().max()
    return (r / (A.abs().sum(1).max() * x.abs().max() + b.abs().max()) / (N * torch.finfo(torch.float64).eps)).item()

ok = True
for N in (1000, 4096, 16384):                                 # small and odd sizes first
    A, b = problem(N, seed=N)
    x = mxp.solve(A, b)
    s = scaled_residual(A, x, b)
    good = x.dtype == torch.float64 and x.shape == b.shape and s < 16
    ok &= good
    print(f"N={N:6d}: scaled residual {s:8.3f} ({'ok' if good else 'FAIL'}, HPL requires < 16)")

N = 32768
A, b = problem(N, seed=1)
t = bench(lambda: mxp.solve(A, b), reps=3)
t64 = bench(lambda: torch.linalg.solve(A, b), reps=3)
x = mxp.solve(A, b)
s = scaled_residual(A, x, b)
ok &= s < 16
flops = 2 / 3 * N**3 + 2 * N**2
print(f"N={N}: your solve {t:.1f} ms ({flops / t / 1e9:.1f} TFLOP/s), FP64 solve {t64:.1f} ms ({flops / t64 / 1e9:.1f}), "
      f"speedup {t64 / t:.2f}x, scaled residual {s:.3f}")

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    mxp.solve(A, b)
    torch.cuda.synchronize()
print("Kernels by GPU time:")
for e in sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:6]:
    print(f"  {e.device_time_total / 1e3:8.1f} ms  {e.count:5d} calls  {e.key[:90]}")
print("PASS" if ok else "FAIL")
