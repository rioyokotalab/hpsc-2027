# Pipeline-parallel GPT on 4 GPUs with torch.distributed.pipelining: GPipe vs 1F1B (Exercise 4).
# Run with: torchrun --standalone --nproc_per_node=4 30_pipeline.py
import sys, time, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, Schedule1F1B
from gpt import GPT, Block
dist.init_process_group("nccl"); r, p = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(r)
d, L, h, T, GBATCH = 2048, 32, 16, 1024, 32
class Stage(nn.Module):
    def __init__(self, first, last, nb):
        super().__init__()
        self.first, self.last = first, last
        if first: self.wte, self.wpe = nn.Embedding(50304, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(nb))
        if last: self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, 50304, bias=False)
    def forward(self, x):
        if self.first: x = self.wte(x) + self.wpe(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks: x = b(x)
        if self.last: x = self.head(self.ln(x))
        return x
def loss_fn(logits, y): return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
torch.manual_seed(r)
mod = Stage(r == 0, r == p - 1, L // p).to("cuda", torch.bfloat16)
nparam = sum(q.numel() for q in mod.parameters())
x = torch.randint(0, 50257, (GBATCH, T), device="cuda"); y = torch.randint(0, 50257, (GBATCH, T), device="cuda")
opt = torch.optim.AdamW(mod.parameters(), lr=1e-4)
if r == 0: print(f"pipeline: {L} blocks d={d}, {p} stages of {L//p} blocks, global batch {GBATCH}x{T}", flush=True)
for sname, S in (("GPipe", ScheduleGPipe), ("1F1B", Schedule1F1B)):
    for m in (1, 2, 4, 8, 16, 32):
        if sname == "1F1B" and m < p: continue
        stage = PipelineStage(mod, r, p, torch.device("cuda"))
        sch = S(stage, n_microbatches=m, loss_fn=loss_fn)
        def step():
            if r == 0: sch.step(x)
            elif r == p - 1: sch.step(target=y, losses=[])
            else: sch.step()
            opt.step(); opt.zero_grad(set_to_none=True)
        try:
            for _ in range(2): step()
            torch.cuda.synchronize(); dist.barrier(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            for _ in range(5): step()
            torch.cuda.synchronize(); dist.barrier(); dt = (time.perf_counter() - t0) / 5
            pk = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device="cuda")
            allpk = [torch.zeros_like(pk) for _ in range(p)]; dist.all_gather(allpk, pk)
            if r == 0: print(f"{sname:6s} microbatches {m:2d} (size {GBATCH//m:2d})  step {dt*1e3:8.1f} ms  peak GB per stage " + " ".join(f"{v.item():5.1f}" for v in allpk), flush=True)
        except torch.OutOfMemoryError:
            if r == 0: print(f"{sname} m={m} OOM", flush=True)
            torch.cuda.empty_cache()
        del sch, stage
# one stage alone, no communication: the time of one microbatch through one stage
for m in (1, 2, 4, 8, 16, 32):
    xb = torch.randint(0, 50257, (GBATCH // m, T), device="cuda")
    hin = torch.randn(GBATCH // m, T, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    def fb():
        out = mod(xb if r == 0 else hin)
        (out.float().square().mean()).backward()
    for _ in range(2): fb()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): fb()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 5
    tt = torch.tensor([dt], device="cuda"); al = [torch.zeros_like(tt) for _ in range(p)]; dist.all_gather(al, tt)
    if r == 0: print(f"one microbatch of {GBATCH//m:2d}: forward+backward per stage ms " + " ".join(f"{v.item()*1e3:6.1f}" for v in al), flush=True)
dist.destroy_process_group()
