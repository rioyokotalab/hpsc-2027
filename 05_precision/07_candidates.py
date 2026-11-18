# Lecture 5, Exercise 4: three BF16 matmul kernels an agent might hand you.
# All three take BF16 A (M x K) and B (K x N) and return BF16 C. Which accumulate in FP32?
import torch, triton, triton.language as tl

@triton.jit
def _kernel(a, b, c, M, N, K, V: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = a + rm[:, None] * K + rk[None, :]
    bp = b + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        p = tl.dot(tl.load(ap), tl.load(bp))
        if V == 0:
            acc += p
        elif V == 1:
            acc = (acc + p).to(tl.bfloat16).to(tl.float32)
        else:
            acc += p.to(tl.bfloat16).to(tl.float32)
        ap += BK
        bp += BK * N
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))

def _run(a, b, v):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    _kernel[(M // 128, N // 128)](a, b, c, M, N, K, V=v, BM=128, BN=128, BK=64, num_warps=8, num_stages=3)
    return c

candidate_a = lambda a, b: _run(a, b, 2)
candidate_b = lambda a, b: _run(a, b, 0)
candidate_c = lambda a, b: _run(a, b, 1)
