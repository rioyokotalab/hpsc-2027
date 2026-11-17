# Attention on one H100: three PyTorch operations against fused kernels.
# Run with the Lecture 3 environment: qsub -g <group> job.sh python 20_attention.py
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def bench(fn, reps=10):
    fn(); torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn()
    t1.record(); t1.synchronize()
    return t0.elapsed_time(t1) / reps

def sdpa(q, k, v, backend):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v)

def naive(q, k, v):                        # writes the S x S score matrix to memory
    s = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    return torch.softmax(s, dim=-1) @ v

B, H, D = 4, 16, 128
for S in (1024, 4096, 8192):
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = naive(q.float(), k.float(), v.float())
    flops = 4 * B * H * S * S * D
    rows = [("naive (3 ops)", lambda: naive(q, k, v))]
    for name, be in (("SDPA flash", SDPBackend.FLASH_ATTENTION), ("SDPA cuDNN", SDPBackend.CUDNN_ATTENTION),
                     ("SDPA efficient", SDPBackend.EFFICIENT_ATTENTION)):
        rows.append((name, lambda be=be: sdpa(q, k, v, be)))
    for name, f in rows:
        try:
            err = (f().float() - ref).abs().max().item()
            ms = bench(f)
            print(f"S {S:5}  {name:15} {ms:8.3f} ms  {flops / ms / 1e9:6.1f} TFLOP/s  max error {err:.1e}")
        except RuntimeError as e:
            print(f"S {S:5}  {name:15} not available: {str(e).splitlines()[0][:60]}")
    print(f"S {S:5}  score matrix {B * H * S * S * 2 / 1e9:.2f} GB in BF16")
