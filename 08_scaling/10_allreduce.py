# All-reduce time, algorithm bandwidth and bus bandwidth vs size (Exercise 2).
# python 10_allreduce.py          all 8 GPUs and the subsets below, 16 B to 1 GiB
# python 10_allreduce.py all      all 8 GPUs only (used with NCCL_ALGO and NCCL_IB_HCA)
import sys, time, torch, torch.distributed as dist, dist_env
r, p = dist_env.init()
groups = [("8 GPUs, two nodes (4 + 4)", list(range(p)))]
if sys.argv[1:] != ["all"]:
    groups += [("4 GPUs, one node", [0, 1, 2, 3]), ("2 GPUs, one node", [0, 1]),
               ("2 GPUs, two nodes (1 + 1)", [0, 4]), ("4 GPUs, two nodes (2 + 2)", [0, 1, 4, 5])]
for name, ranks in groups:
    g = dist.new_group(ranks)                     # every rank must call new_group
    if r in ranks:
        q = len(ranks); f = 2 * (q - 1) / q
        if r == 0: print(f"# {name}", flush=True)
        for n in [2**k for k in range(4, 31)]:
            x = torch.ones(n // 2, device="cuda", dtype=torch.bfloat16)
            reps = 100 if n < 2**24 else 10
            for _ in range(5): dist.all_reduce(x, group=g)
            torch.cuda.synchronize(); dist.barrier(g); t0 = time.perf_counter()
            for _ in range(reps): dist.all_reduce(x, group=g)
            torch.cuda.synchronize(); t = (time.perf_counter() - t0) / reps
            if r == 0: print(f"allreduce p={q} {n:11d} bytes {t*1e6:10.2f} us algbw {n/t/1e9:7.2f} busbw {n/t/1e9*f:7.2f} GB/s", flush=True)
        del x; torch.cuda.empty_cache()
    dist.barrier()
dist.destroy_process_group()
