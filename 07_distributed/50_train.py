# Homework template: train a 1.5B-parameter GPT (d 1600, 48 layers) for STEPS steps with a
# global batch of 32 sequences of 1024 tokens on the 4 H100s of one node.
# Run with: torchrun --standalone --nproc_per_node=4 50_train.py
# Each GPU takes 8 of the 32 sequences, one at a time (8 micro-batches), and the model is
# sharded with FSDP (fully_shard). Make it faster without changing what is computed: same
# data, same global batch, same model, same optimizer, same number of steps.
import os, time, torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from gpt import GPT

STEPS, GLOBAL_BATCH, T = 60, 32, 1024
MICRO = 1                                          # sequences per micro-batch on each GPU
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


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
with torch.device("meta"):
    model = GPT(T=T, d=1600, L=48, h=25)
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.blocks:
    fully_shard(block, mp_policy=mp)               # each GPU keeps 1/4 of every block
fully_shard(model, mp_policy=mp)
model.to_empty(device="cuda")
model.init_weights()                               # same initial weights on every GPU count (seeded)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

local = GLOBAL_BATCH // world                      # sequences per GPU per step
dist.barrier(); torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(STEPS):
    batch = data[step, rank * local:(rank + 1) * local].cuda(non_blocking=True)
    x, y = batch[:, :-1], batch[:, 1:]
    loss_sum = torch.zeros((), device="cuda")
    for mb in range(0, local, MICRO):
        loss = model(x[mb:mb + MICRO], y[mb:mb + MICRO]) * MICRO / local
        loss.backward()
        loss_sum += loss.detach()
    opt.step()
    opt.zero_grad(set_to_none=True)
    dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)            # mean over all 32 sequences
    if rank == 0 and (step % 5 == 0 or step == STEPS - 1):
        print(f"step {step:4d} loss {loss_sum.item():.4f}", flush=True)
torch.cuda.synchronize()
if rank == 0:
    print(f"train time {time.perf_counter() - t0:.2f} s")
    print(f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB on GPU 0")
dist.destroy_process_group()
