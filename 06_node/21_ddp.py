# Data-parallel training of the 124M GPT on the GPUs of one node.
# Run with: torchrun --standalone --nproc_per_node=4 21_ddp.py [compile]
# Three settings: no communication (DDP.no_sync, compute only), DDP with
# 25 MB buckets (all-reduce overlaps the backward pass), and DDP with one
# bucket (the all-reduce starts after the backward pass).
import os, sys, time
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, PEAK

T = 1024
dist.init_process_group("nccl")
rank, p = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)


def run(name, B, bucket_mb, sync=True, compile=False, steps=20):
    torch.manual_seed(0)
    m = GPT(T=T).cuda()
    dm = DDP(m, device_ids=[rank], bucket_cap_mb=bucket_mb)
    if compile:
        dm = torch.compile(dm)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
    g = torch.Generator(device="cuda").manual_seed(1 + rank)          # each rank gets different data
    x = torch.randint(0, 50257, (B, T), device="cuda", generator=g)
    y = torch.randint(0, 50257, (B, T), device="cuda", generator=g)

    def step():
        if sync:
            with torch.autocast("cuda", torch.bfloat16):
                loss = dm(x, y)
            loss.backward()
        else:
            with dm.no_sync():                                        # skip the all-reduce
                with torch.autocast("cuda", torch.bfloat16):
                    loss = dm(x, y)
                loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    if rank == 0:
        tok = p * B * T / dt
        print(f"batch {B:2d}  {name:28s} step {dt * 1e3:6.1f} ms  {tok:8.0f} tokens/s  MFU {100 * tok * m.flops_per_token() / (p * PEAK):5.1f} %",
              flush=True)


if rank == 0:
    print(f"{p} GPUs, batch B x {T} tokens per GPU, gradients {124.5e6 * 4 / 1e6:.0f} MB in FP32")
for B in (16, 4, 1):
    run("compute only (no all-reduce)", B, 25, sync=False)
    run("DDP, 25 MB buckets", B, 25)
    run("DDP, one bucket", B, 1000)
if len(sys.argv) > 1 and sys.argv[1] == "compile":          # optional, about 1 more minute
    run("compiled, compute only", 16, 25, sync=False, compile=True)
    run("compiled, DDP, 25 MB buckets", 16, 25, compile=True)
dist.destroy_process_group()
