# Copy 1 GiB between GPUs (NVLink) and from pinned host memory to a GPU (PCIe).
import time, torch

n = 2**30


def bw(dst, src, sync):
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    for d in sync:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(10):
        dst.copy_(src, non_blocking=True)
    for d in sync:
        torch.cuda.synchronize(d)
    return 10 * n / (time.perf_counter() - t0) / 1e9


g = torch.cuda.device_count()
for s in range(g):
    a = torch.empty(n, dtype=torch.uint8, device=f"cuda:{s}")
    line = []
    for d in range(g):
        if d == s:
            line.append("    -  ")
            continue
        b = torch.empty(n, dtype=torch.uint8, device=f"cuda:{d}")
        line.append(f"{bw(b, a, (s, d)):6.1f} ")
        del b
    print(f"GPU {s} -> GPU 0..{g - 1} (GB/s): " + " ".join(line), flush=True)
h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
print(f"pinned host -> GPU 0: {bw(torch.empty(n, dtype=torch.uint8, device='cuda:0'), h, (0,)):.1f} GB/s")
print(f"GPU 0 -> pinned host: {bw(h, torch.empty(n, dtype=torch.uint8, device='cuda:0'), (0,)):.1f} GB/s")
