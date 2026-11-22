# Ring attention (context parallelism): each GPU keeps one block of queries; keys and values travel around the ring.
# Run with: torchrun --standalone --nproc_per_node=4 40_ring_attention.py
import time, torch, torch.distributed as dist, torch.nn.functional as F
dist.init_process_group("nccl"); r, p = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(r)
H, D = 16, 128
def ring_attention(q, k, v):
    # q, k, v: (H, S/p, D) on this GPU. Online softmax over the visiting blocks.
    o = torch.zeros_like(q, dtype=torch.float32)
    m = torch.full(q.shape[:2], -float("inf"), device=q.device)      # running max per query
    l = torch.zeros(q.shape[:2], device=q.device)                     # running sum of exp
    for step in range(p):
        if step < p - 1:                                              # start sending k, v to the next GPU
            kn, vn = torch.empty_like(k), torch.empty_like(v)
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, k, (r + 1) % p), dist.P2POp(dist.irecv, kn, (r - 1) % p),
                                           dist.P2POp(dist.isend, v, (r + 1) % p), dist.P2POp(dist.irecv, vn, (r - 1) % p)])
        s = (q @ k.transpose(1, 2)).float() / D**0.5                  # (H, S/p, S/p)
        mn = torch.maximum(m, s.amax(-1))
        a = torch.exp(m - mn); e = torch.exp(s - mn[..., None])
        l = l * a + e.sum(-1); o = o * a[..., None] + e.to(v.dtype) @ v
        m = mn
        if step < p - 1:
            for q_ in reqs: q_.wait()
            k, v = kn, vn
    return (o / l[..., None]).to(q.dtype)
def bench(f, reps=5, sync=True):
    f(); torch.cuda.synchronize()
    if sync: dist.barrier()
    t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps
for S in (8192, 32768, 65536):
    torch.manual_seed(0)
    Q, K, V = (torch.randn(H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    n = S // p
    q, k, v = Q[:, r*n:(r+1)*n].contiguous(), K[:, r*n:(r+1)*n].contiguous(), V[:, r*n:(r+1)*n].contiguous()
    torch.cuda.reset_peak_memory_stats()
    out = ring_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q[None], K[None], V[None])[0]
    err = (out.float() - ref.float()).abs().max().item()
    tr = bench(lambda: ring_attention(q, k, v)); pk = torch.cuda.max_memory_allocated() / 1e9
    t1 = bench(lambda: F.scaled_dot_product_attention(Q[None], K[None], V[None]), sync=False) if r == 0 else 0.0
    dist.barrier()
    if r == 0:
        print(f"S {S:6d}: ring on {p} GPUs {tr*1e3:8.2f} ms (peak {pk:5.1f} GB)  SDPA on 1 GPU {t1*1e3:8.2f} ms  max|diff| {err:.2e}  K,V block {2*H*n*D*2/1e6:.0f} MB per step", flush=True)
dist.destroy_process_group()
