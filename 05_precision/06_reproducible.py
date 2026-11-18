# Lecture 5: run-to-run differences (atomics) and batch invariance.
import torch
from bench import bench

torch.manual_seed(0)
x = torch.randn(10_000_000, device="cuda")
idx = torch.randint(0, 1000, (10_000_000,), device="cuda")
def scatter():
    return torch.zeros(1000, device="cuda").index_add_(0, idx, x)

for det in (False, True):
    torch.use_deterministic_algorithms(det)
    results = {scatter().cpu().numpy().tobytes() for _ in range(20)}
    print(f"index_add_ deterministic={det!s:5s}: {len(results):2d} different results in 20 runs, {bench(scatter):.3f} ms")
torch.use_deterministic_algorithms(False)

results = {x.sum().item() for _ in range(20)}
print(f"torch.sum: {len(results)} different result(s) in 20 runs")

# Batch invariance: is row 0 of X @ W the same when X has more rows?
torch.backends.cuda.matmul.allow_tf32 = False
print("\nRow 0 of X[:B] @ W compared with B = 1 (fraction of the row's bits that differ)")
for dt in (torch.float32, torch.bfloat16):
    for K, N in ((4096, 4096), (8192, 1024), (16384, 512)):
        W = torch.randn(K, N, device="cuda", dtype=dt)
        X = torch.randn(4096, K, device="cuda", dtype=dt)
        r1 = X[:1] @ W
        diff = [((X[:B] @ W)[:1] != r1).float().mean().item() for B in (1, 8, 64, 512, 4096)]
        print(f"  {str(dt)[6:]:8s} K={K:5d} N={N:5d}  B = 1, 8, 64, 512, 4096: " + "  ".join(f"{d:.2f}" for d in diff))
