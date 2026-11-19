# Homework template: train the 124M GPT for STEPS steps with a global batch of
# 32 sequences of 1024 tokens. It uses one GPU. Make it faster on one node
# (4 H100s) without changing what is computed: same data, same global batch,
# same model, same optimizer, same number of steps.
# check_train.py runs this file with torchrun on 4 GPUs and compares the losses.
import os, time, torch
from gpt import GPT

STEPS, GLOBAL_BATCH, T = 300, 32, 1024
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)


def make_data(steps, batch, T, vocab=50257, seed=0):
    # A synthetic token stream: each token has 4 possible successors (loss can fall to ln 4)
    g = torch.Generator().manual_seed(seed)
    succ = torch.randint(0, vocab, (vocab, 4), generator=g)
    tok = torch.empty(steps * batch, T + 1, dtype=torch.long)
    tok[:, 0] = torch.randint(0, vocab, (steps * batch,), generator=g)
    choice = torch.randint(0, 4, (steps * batch, T), generator=g)
    for t in range(T):
        tok[:, t + 1] = succ[tok[:, t], choice[:, t]]
    return tok.view(steps, batch, T + 1)


data = make_data(STEPS, GLOBAL_BATCH, T)
torch.manual_seed(0)
model = GPT(T=T).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, fused=True)

t0 = time.perf_counter()
for step in range(STEPS):
    batch = data[step].cuda(non_blocking=True)
    x, y = batch[:, :-1], batch[:, 1:]
    loss_sum = 0.0
    for mb in range(0, GLOBAL_BATCH, 16):                     # two micro-batches of 16 fit in memory
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(x[mb:mb + 16], y[mb:mb + 16]) * 16 / GLOBAL_BATCH
        loss.backward()
        loss_sum += loss.detach()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if rank == 0 and (step % 10 == 0 or step == STEPS - 1):
        print(f"step {step:4d} loss {loss_sum.item():.4f}", flush=True)
torch.cuda.synchronize()
if rank == 0:
    print(f"train time {time.perf_counter() - t0:.2f} s")
