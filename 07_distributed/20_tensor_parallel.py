# Tensor-parallel MLP (Megatron-LM): W1 split by columns, W2 by rows, one all-reduce (Exercise 3).
# Run with: torchrun --standalone --nproc_per_node=4 20_tensor_parallel.py
import sys, time, torch, torch.distributed as dist, torch.nn.functional as F
dist.init_process_group("nccl"); r, p = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(r)
def bench(f, reps=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps
for d in (2048, 4096, 8192):
    for tok in (1024, 8192):
        torch.manual_seed(0)
        X = torch.randn(tok, d, device="cuda", dtype=torch.bfloat16)
        W1 = torch.randn(d, 4 * d, device="cuda", dtype=torch.bfloat16) / d**0.5
        W2 = torch.randn(4 * d, d, device="cuda", dtype=torch.bfloat16) / (4 * d)**0.5
        k = 4 * d // p
        W1r, W2r = W1[:, r*k:(r+1)*k].contiguous(), W2[r*k:(r+1)*k, :].contiguous()
        def full(): return F.gelu(X @ W1) @ W2
        def part(): return F.gelu(X @ W1r) @ W2r
        def tp():
            Z = part(); dist.all_reduce(Z); return Z
        err = (tp() - full()).abs().max().item(); ref = full().abs().max().item()
        Z = part(); t1, tpart, ttp = bench(full), bench(part), bench(tp)
        tar = bench(lambda: dist.all_reduce(Z))
        if r == 0:
            print(f"d {d:5d} tokens {tok:5d}  1 GPU {t1*1e3:7.3f} ms  TP{p} compute {tpart*1e3:7.3f} ms  all-reduce {tar*1e3:7.3f} ms  TP{p} total {ttp*1e3:7.3f} ms  speed-up {t1/ttp:4.2f}  max|diff| {err:.3g} (max|Z| {ref:.3g})  AR bytes {Z.numel()*2/1e6:.1f} MB", flush=True)
# the error you get if you average instead of sum
Z = part(); dist.all_reduce(Z, op=dist.ReduceOp.AVG)
if r == 0: print(f"AVG instead of SUM: ratio to full {(Z.float().norm()/full().float().norm()).item():.3f}", flush=True)
dist.destroy_process_group()
