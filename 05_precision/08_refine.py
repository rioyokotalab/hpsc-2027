# Lecture 5: mixed-precision iterative refinement: FP32 LU, FP64 residual.
import torch
from bench import bench

torch.manual_seed(0)
N = 16384
A = torch.randn(N, N, device="cuda", dtype=torch.float64)
b = torch.randn(N, 1, device="cuda", dtype=torch.float64)
print(f"N={N}: LU in FP64 {bench(lambda: torch.linalg.lu_factor(A), 2):.1f} ms, "
      f"in FP32 {bench(lambda: torch.linalg.lu_factor(A.float()), 2):.1f} ms")
x64 = torch.linalg.solve(A, b)
LU, piv = torch.linalg.lu_factor(A.float())                        # O(N^3) in FP32
x = torch.linalg.lu_solve(LU, piv, b.float()).double()
for it in range(6):
    r = b - A @ x                                                  # O(N^2) in FP64
    berr = (r.norm() / (A.norm() * x.norm() + b.norm())).item()
    print(f"iteration {it}: backward error {berr:.1e}, difference to FP64 solve {((x - x64).norm() / x64.norm()).item():.1e}")
    x = x + torch.linalg.lu_solve(LU, piv, r.float()).double()
r = b - A @ x64
print(f"FP64 solve:  backward error {(r.norm() / (A.norm() * x64.norm() + b.norm())).item():.1e}")
