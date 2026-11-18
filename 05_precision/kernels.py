# Lecture 5: a BF16 matrix multiplication in Triton whose accumulator precision you choose.
import torch, triton, triton.language as tl

@triton.jit
def matmul_kernel(a, b, c, M, N, K, ACC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = a + rm[:, None] * K + rk[None, :]
    bp = b + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        p = tl.dot(tl.load(ap), tl.load(bp))                 # BK-deep product on the tensor cores
        if ACC == "fp32":
            acc += p                                         # FP32 accumulator
        else:
            acc = (acc + p).to(tl.bfloat16).to(tl.float32)   # accumulator rounded to BF16 every BK
        ap += BK
        bp += BK * N
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))

def matmul(a, b, acc="fp32"):
    """C = A @ B for BF16 A (M x K), B (K x N); M, N multiples of 128, K of 64."""
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    matmul_kernel[(M // 128, N // 128)](a, b, c, M, N, K, ACC=acc, BM=128, BN=128, BK=64, num_warps=8, num_stages=3)
    return c
