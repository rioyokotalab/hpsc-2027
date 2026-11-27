# DDP training step of a GPT: total step, the step without gradient all-reduce (no_sync),
# and one all-reduce of all FP32 gradients, for several batch sizes per GPU (Exercise 3).
# python 20_scaling.py d L h B1 B2 ...   (B = sequences of 1,024 tokens per GPU)
import sys, time, torch, torch.distributed as dist, dist_env
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, PEAK
r, p = dist_env.init()
d, L, h = map(int, sys.argv[1:4]); T = 1024
torch.manual_seed(0)
m = GPT(T=T, d=d, L=L, h=h).cuda()
n = sum(q.numel() for q in m.parameters())
model = DDP(m, device_ids=[torch.cuda.current_device()], bucket_cap_mb=25, gradient_as_bucket_view=True)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, foreach=True)
grad = torch.empty(n, device="cuda", dtype=torch.float32)
for _ in range(3): dist.all_reduce(grad)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(5): dist.all_reduce(grad)
torch.cuda.synchronize(); ta = (time.perf_counter() - t0) / 5
del grad; torch.cuda.empty_cache()
for B in map(int, sys.argv[4:]):
    g = torch.Generator(device="cuda").manual_seed(1 + r)
    x = torch.randint(0, 50257, (B, T), device="cuda", generator=g); y = torch.randint(0, 50257, (B, T), device="cuda", generator=g)
    def step(sync):
        with torch.autocast("cuda", torch.bfloat16):
            if sync: loss = model(x, y); loss.backward()
            else:
                with model.no_sync():
                    loss = model(x, y); loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    def timed(sync, k=8):
        for _ in range(3): step(sync)
        torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
        for _ in range(k): step(sync)
        torch.cuda.synchronize(); t = torch.tensor((time.perf_counter() - t0) / k, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX); return t.item()
    ts, tn = timed(True), timed(False)
    if r == 0:
        tok = p * B * T / ts
        print(f"scaling p={p} params {n/1e6:.0f} M batch {B}/GPU ({p*B} total): step {ts*1e3:8.1f} ms  no_sync {tn*1e3:8.1f} ms  "
              f"allreduce(fp32 grads {4*n/1e9:.2f} GB) {ta*1e3:7.1f} ms  tokens/s {tok:10.0f}  MFU {100*tok*m.flops_per_token()/(p*PEAK):5.1f} %", flush=True)
    del x, y; torch.cuda.empty_cache()
dist.destroy_process_group()
