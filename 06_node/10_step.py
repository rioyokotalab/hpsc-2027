# One training step of a 124M-parameter GPT on one H100: eager, eager with
# activation checkpointing, and torch.compile. Prints time, MFU, memory and
# where the GPU time goes.
import os, time
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"   # measure a cold compile every time
import torch
from torch.profiler import profile, ProfilerActivity
from gpt import GPT, PEAK

B, T = 16, 1024


def kind(name):
    n = name.lower()
    if "flash" in n or "fmha" in n or "attention" in n:
        return "attention"
    if any(k in n for k in ("gemm", "nvjet", "xmma", "cutlass", "sm90")):
        return "matmul"
    return "other"


def run(mode, ckpt=False, steps=20):
    torch.manual_seed(0)
    m = GPT(T=T, ckpt=ckpt).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
    f = torch.compile(m) if mode == "compile" else m
    x = torch.randint(0, 50257, (B, T), device="cuda")
    y = torch.randint(0, 50257, (B, T), device="cuda")

    def step():
        with torch.autocast("cuda", torch.bfloat16):
            loss = f(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss0 = step().item()                      # the first step includes compilation
    first = time.perf_counter() - t0
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    mem = torch.cuda.max_memory_allocated() / 2**30
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    split, nk = {"matmul": 0.0, "attention": 0.0, "other": 0.0}, 0
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            split[kind(e.key)] += e.self_device_time_total / 1e3   # ms
            nk += e.count
    tok = B * T / dt
    print(f"{mode + (' + checkpoint' if ckpt else ''):20s} first step {first:6.1f} s  step {dt * 1e3:6.1f} ms  "
          f"{tok:8.0f} tokens/s  MFU {100 * tok * m.flops_per_token() / PEAK:5.1f} %  "
          f"peak memory {mem:5.1f} GiB  loss {loss0:.4f}")
    print(f"{'':20s} {nk} kernels per step; GPU time (ms): " +
          ", ".join(f"{k} {v:.1f}" for k, v in split.items()), flush=True)


m = GPT(T=T)
print(f"GPT: {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M parameters, "
      f"{m.flops_per_token() / 1e6:.0f} MFLOP per token, batch {B} x {T} tokens")
del m
run("eager")
run("eager", ckpt=True)
run("compile")
