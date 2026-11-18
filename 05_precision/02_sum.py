# Lecture 5: the error of a sum depends on the order of the additions.
import math, numpy as np, torch

rng = np.random.default_rng(0)
print(f"{'n':>11s} {'sequential FP32':>16s} {'NumPy (pairwise)':>17s} {'torch.sum on GPU':>17s}")
for e in range(3, 9):
    n = 10**e
    x = rng.random(n, dtype=np.float32)
    exact = math.fsum(x.tolist()) if n <= 10**7 else float(np.sum(x, dtype=np.float64))
    seq = np.cumsum(x, dtype=np.float32)[-1]                 # one addition after another
    pair = np.sum(x, dtype=np.float32)                       # NumPy sums in pairs
    gpu = torch.from_numpy(x).cuda().sum().item()            # a tree of partial sums
    r = lambda s: abs(float(s) - exact) / exact
    print(f"{n:11d} {r(seq):16.2e} {r(pair):17.2e} {r(gpu):17.2e}")
