# Collective communication on the GPUs of one node: time and bus bandwidth vs message size (Exercise 1).
# Run with: torchrun --standalone --nproc_per_node=4 00_collectives.py
import time, sys, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, p = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
F = {"all_reduce": 2 * (p - 1) / p, "reduce_scatter": (p - 1) / p, "all_gather": (p - 1) / p,
     "all_to_all": (p - 1) / p, "broadcast": 1.0, "sendrecv": 1.0}
def op(name, x, y):
    if name == "all_reduce": dist.all_reduce(x)
    elif name == "reduce_scatter": dist.reduce_scatter_tensor(y, x)
    elif name == "all_gather": dist.all_gather_into_tensor(x, y)
    elif name == "all_to_all": dist.all_to_all_single(y.new_empty(x.shape), x)
    elif name == "broadcast": dist.broadcast(x, 0)
    elif name == "sendrecv":   # ring shift: send to r+1, receive from r-1
        ops = [dist.P2POp(dist.isend, x, (r + 1) % p), dist.P2POp(dist.irecv, x.new_empty(x.shape), (r - 1) % p)]
        for q in dist.batch_isend_irecv(ops): q.wait()
for name in F:
    if r == 0: print(f"--- {name}", flush=True)
    for n in [2**k for k in range(16, 31, 2)]:
        x = torch.ones(n // 2, device="cuda", dtype=torch.bfloat16)   # n bytes = the full tensor
        y = torch.ones(n // 2 // p, device="cuda", dtype=torch.bfloat16)
        for _ in range(5): op(name, x, y)
        torch.cuda.synchronize(); dist.barrier()
        reps = 40 if n < 2**26 else 10
        t0 = time.perf_counter()
        for _ in range(reps): op(name, x, y)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / reps
        if r == 0:
            print(f"{name:15s} {n:11d} bytes {t*1e6:9.1f} us algbw {n/t/1e9:7.1f} GB/s busbw {n/t/1e9*F[name]:7.1f} GB/s", flush=True)
dist.destroy_process_group()
