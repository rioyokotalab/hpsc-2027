# Lecture 5: floating-point formats, unit roundoff and stagnation (Exercise 1).
import torch

dev = "cuda"
print(f"{'format':10s} {'bits':>4s} {'unit roundoff u':>16s} {'smallest normal':>16s} {'largest':>12s}")
for dt in (torch.float64, torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2):
    f = torch.finfo(dt)
    print(f"{str(dt)[6:]:10s} {f.bits:4d} {f.eps / 2:16.3e} {f.tiny:16.3e} {f.max:12.6g}")

print("\nOne rounding: 1 + 2^-9 and 1 + 1/1024")
for dt in (torch.bfloat16, torch.float16, torch.float32):
    one = torch.ones((), dtype=dt, device=dev)
    print(f"{str(dt)[6:]:9s} 1 + 2^-9 = {(one + 2**-9).item():.10f}   1 + 1/1024 = {(one + 1 / 1024).item():.10f}")

print("\nStagnation: add 1 to a counter until it stops changing")
for dt, start in ((torch.bfloat16, 0), (torch.float16, 0), (torch.float32, 2**24 - 8)):
    s = torch.full((), start, dtype=dt, device=dev)
    one = torch.ones((), dtype=dt, device=dev)
    while (s + one) != s:
        s = s + one
    print(f"{str(dt)[6:]:9s} counter stops at {s.item():.0f}")

print("\nStagnation of an update: w = 1, then w += 0.001 one thousand times")
for dt in (torch.bfloat16, torch.float16, torch.float32):
    w = torch.ones((), dtype=dt, device=dev)
    for _ in range(1000):
        w = w + 0.001
    print(f"{str(dt)[6:]:9s} w = {w.item():.6f}   (exact: 2)")
