import importlib
import torch
from torch.profiler import profile, ProfilerActivity
from bench import bench

silu_mul = importlib.import_module("08_silu_mul").silu_mul
ref = lambda a, b: (torch.nn.functional.silu(a.double()) * b.double())
ok = True

def kernels(a, b):                         # names of the GPU kernels one call launches
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        silu_mul(a, b)
        torch.cuda.synchronize()
    return {e.name for e in p.events() if e.device_type.name == "CUDA"}

for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
    for M, N in ((1, 1), (7, 13), (1000, 1001), (4096, 11008)):
        for strided in (False, True):
            g = torch.Generator(device="cuda").manual_seed(M * N)
            a = torch.randn(M, 2 * N if strided else N, device="cuda", generator=g).to(dtype)
            b = torch.randn(M, 2 * N if strided else N, device="cuda", generator=g).to(dtype)
            if strided:
                a, b = a[:, ::2], b[:, ::2]
            c = silu_mul(a, b)
            err = ((c.double() - ref(a, b)).abs() / (1 + ref(a, b).abs())).max().item()
            good = c.shape == a.shape and c.dtype == dtype and c.is_contiguous() and err <= tol
            ok &= good
            print(f"{str(dtype):15} {M:5} x {N:5} {'strided' if strided else 'contig.':8} "
                  f"error {err:.1e}  {'ok' if good else 'FAIL'}")

a = torch.randn(4096, 11008, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
names = kernels(a, b)
others = {n for n in names if "at::" in n}    # kernels from PyTorch, not Triton
print("GPU kernels:", ", ".join(sorted(names)))
if others:
    ok = False
    print("FAIL: PyTorch kernels ran:", ", ".join(sorted(others)))
t = bench(lambda: silu_mul(a, b)) / 1e3
print(f"bfloat16 4096 x 11008: {t*1e3:.3f} ms  {3 * a.numel() * 2 / t / 1e9:.0f} GB/s")
print("PASS" if ok else "FAIL")
