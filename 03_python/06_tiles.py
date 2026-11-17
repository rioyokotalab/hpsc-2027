import torch
from bench import bench
import importlib
mm = importlib.import_module("05_matmul")

N = 4096
a = torch.randn(N, N, device="cuda").bfloat16()
b = torch.randn(N, N, device="cuda").bfloat16()
print(" BM  BN  BK warps stages  regs spills  smem(KB)  TFLOP/s")
for BM, BN, BK, w, s in ((32, 32, 32, 4, 3), (64, 64, 32, 4, 3), (128, 128, 32, 4, 3),
                         (128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 256, 64, 8, 3),
                         (128, 256, 64, 8, 4), (256, 256, 64, 8, 3)):
    try:
        k, _ = mm.matmul(a, b, "ieee", BM, BN, BK, w, s)
        ms = bench(lambda: mm.matmul(a, b, "ieee", BM, BN, BK, w, s))
        print(f"{BM:3} {BN:3} {BK:3} {w:5} {s:6} {k.n_regs:5} {k.n_spills:6} "
              f"{k.metadata.shared/1024:9.0f}  {2*N**3/(ms/1e3)/1e12:7.1f}")
    except Exception as e:                 # e.g. out of shared memory
        print(f"{BM:3} {BN:3} {BK:3} {w:5} {s:6}  failed: {type(e).__name__}")
