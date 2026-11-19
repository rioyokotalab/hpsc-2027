# All-reduce of n bytes across the GPUs of one node with NCCL.
# Run with: torchrun --standalone --nproc_per_node=4 20_allreduce.py
import time, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank, p = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
if rank == 0:
    print(f"all-reduce (sum, FP32) on {p} GPUs; busbw = algbw x 2(p-1)/p")
for n in [2**k for k in range(12, 31, 2)]:
    x = torch.ones(n // 4, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    reps = 50 if n < 2**24 else 10
    t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / reps
    if rank == 0:
        algbw = n / t / 1e9
        print(f"{n:11d} bytes  {t * 1e6:9.1f} us  algbw {algbw:6.1f} GB/s  busbw {algbw * 2 * (p - 1) / p:6.1f} GB/s",
              flush=True)
dist.destroy_process_group()
