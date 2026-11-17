import torch

# Homework: replace this function with a Triton kernel for RMSNorm, the
# normalisation layer of Llama-type models: y = x / sqrt(mean(x^2) + eps) * w,
# the mean taken over each row.
# Contract: x is (M, N), float32 or bfloat16, and may be non-contiguous;
# w is (N,) with the same dtype; accumulate in float32; return a new contiguous
# tensor with the dtype of x; only Triton kernels may run on the GPU.
# Then run: python check_rmsnorm.py
def rmsnorm(x, w, eps=1e-6):
    return torch.nn.functional.rms_norm(x, (x.shape[1],), w, eps)
