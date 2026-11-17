# Run under Nsight Compute: ./profile.sh python 09_profile.py
import importlib
import torch
sm = importlib.import_module("04_softmax")
mm = importlib.import_module("05_matmul")

x = torch.randn(8192, 4096, device="cuda")
a = torch.randn(4096, 4096, device="cuda").bfloat16()
cases = [lambda: sm.softmax_naive(x),                            # five kernels
         lambda: sm.softmax_triton(x),                           # one kernel
         lambda: mm.matmul(a, a, "ieee", 128, 256, 64, 8, 4),    # fast tile
         lambda: mm.matmul(a, a, "ieee", 256, 256, 64, 8, 3)]    # registers spill
for f in cases:                            # compile and warm up outside the profile
    f()
torch.cuda.synchronize()
torch.cuda.profiler.start()                # ncu records only from here
for f in cases:
    f()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
