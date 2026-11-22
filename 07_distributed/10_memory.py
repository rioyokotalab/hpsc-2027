# Memory per GPU of DDP, ZeRO-1 and FSDP for GPTs of different sizes (Exercise 2).
# Run with: torchrun --standalone --nproc_per_node=4 10_memory.py {ddp|zero1|fsdp|fsdp_noreshard} d L h B
import os, sys, time, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, PEAK
mode, d, L, h, B = sys.argv[1], *map(int, sys.argv[2:6])
T = 1024
dist.init_process_group("nccl"); r, p = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(r)
GB = 1e9
def mem(tag):
    torch.cuda.synchronize()
    if r == 0:
        print(f"  {tag:22s} allocated {torch.cuda.memory_allocated()/GB:6.2f} GB  peak {torch.cuda.max_memory_allocated()/GB:6.2f} GB", flush=True)
torch.manual_seed(0)
with torch.device("meta"):
    m = GPT(T=T, d=d, L=L, h=h)
n = sum(q.numel() for q in m.parameters())
if r == 0: print(f"=== {mode}  d={d} L={L} params {n/1e9:.3f} B  batch {B}x{T} per GPU, {p} GPUs", flush=True)
try:
    if mode.startswith("fsdp"):
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        rs = mode == "fsdp"
        for b in m.blocks: fully_shard(b, mp_policy=mp, reshard_after_forward=rs)
        fully_shard(m, mp_policy=mp, reshard_after_forward=rs)
        m.to_empty(device="cuda"); m.init_weights(); model = m; auto = False
    else:
        m.to_empty(device="cuda"); m.init_weights()
        model = DDP(m, device_ids=[r], bucket_cap_mb=25, gradient_as_bucket_view=True); auto = True
    mem("model")
    if mode == "zero1":
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(m.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-4)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, foreach=True)
    g = torch.Generator(device="cuda").manual_seed(1 + r)
    x = torch.randint(0, 50257, (B, T), device="cuda", generator=g); y = torch.randint(0, 50257, (B, T), device="cuda", generator=g)
    def step(tag=None):
        if auto:
            with torch.autocast("cuda", torch.bfloat16): loss = model(x, y)
        else:
            loss = model(x, y)
        if tag: mem(tag + " forward")
        loss.backward()
        if tag: mem(tag + " backward")
        opt.step(); opt.zero_grad(set_to_none=True)
        if tag: mem(tag + " optimizer")
        return loss
    step("step 1")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2): step()
    torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
    for _ in range(5): loss = step()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 5
    mem("steady state")
    if r == 0:
        tok = p * B * T / dt
        print(f"  step {dt*1e3:7.1f} ms  MFU {100*tok*m.flops_per_token()/(p*PEAK):5.1f} %  loss {loss.item():.3f}", flush=True)
except torch.OutOfMemoryError as e:
    if r == 0: print(f"  OUT OF MEMORY ({str(e).splitlines()[0][:90]})", flush=True); mem("at OOM")
    os._exit(0)
dist.destroy_process_group()
