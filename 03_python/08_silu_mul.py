import torch

# Exercise 4: ask your coding agent to replace this function with a Triton
# kernel that computes silu(a) * b, the gate of a SwiGLU MLP.
# Contract: a and b have the same shape (M, N) and dtype (float32 or bfloat16),
# may be non-contiguous; return a new contiguous tensor; only Triton kernels
# may run on the GPU. Then run: python check_silu_mul.py
def silu_mul(a, b):
    return torch.nn.functional.silu(a) * b
