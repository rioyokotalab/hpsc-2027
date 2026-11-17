import importlib
import torch
from torch.profiler import profile, ProfilerActivity
from bench import bench

rmsnorm = importlib.import_module("10_rmsnorm").rmsnorm
eps = 1e-6
def ref(x, w):
    x, w = x.double(), w.double()
    return x / torch.sqrt((x * x).mean(dim=1, keepdim=True) + eps) * w
ok = True

def kernels(x, w):                         # names of the GPU kernels one call launches
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        rmsnorm(x, w, eps)
        torch.cuda.synchronize()
    return {e.name for e in p.events() if e.device_type.name == "CUDA"}

for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
    for M, N in ((1, 1), (7, 13), (1000, 1001), (4096, 4096)):
        for strided in (False, True):
            g = torch.Generator(device="cuda").manual_seed(M * N)
            x = torch.randn(M, 2 * N if strided else N, device="cuda", generator=g).to(dtype)
            w = torch.randn(N, device="cuda", generator=g).to(dtype)
            if strided:
                x = x[:, ::2]
            y = rmsnorm(x, w, eps)
            err = ((y.double() - ref(x, w)).abs() / (1 + ref(x, w).abs())).max().item()
            good = y.shape == x.shape and y.dtype == dtype and y.is_contiguous() and err <= tol
            ok &= good
            print(f"{str(dtype):15} {M:5} x {N:5} {'strided' if strided else 'contig.':8} "
                  f"error {err:.1e}  {'ok' if good else 'FAIL'}")

M, N = 16384, 4096                         # 16,384 tokens of a model with hidden size 4,096
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
names = kernels(x, w)
others = {n for n in names if "at::" in n}    # kernels from PyTorch, not Triton
print("GPU kernels:", ", ".join(sorted(names)))
if others:
    ok = False
    print("FAIL: PyTorch kernels ran:", ", ".join(sorted(others)))
for name, f in (("yours", lambda: rmsnorm(x, w, eps)),
                ("torch rms_norm", lambda: torch.nn.functional.rms_norm(x, (N,), w, eps))):
    t = bench(f) / 1e3
    print(f"bfloat16 {M} x {N} {name:15} {t*1e3:.3f} ms  {2 * x.numel() * 2 / t / 1e9:.0f} GB/s")
print("PASS" if ok else "FAIL")
