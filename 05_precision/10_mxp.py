# Lecture 5 homework: HPL-MxP on one GPU.
# Solve A x = b to FP64 accuracy, faster than an FP64 solve, by factorizing A in
# lower precision on the tensor cores and refining x with FP64 residuals.
# A is diagonally dominant (as in HPL-MxP), so LU without pivoting is stable.
# Replace this FP64 baseline; keep the signature. check_mxp.py tests and times it.
import torch

def solve(A, b):
    """A: N x N FP64 on the GPU, b: N x 1 FP64. Return x (N x 1, FP64)."""
    return torch.linalg.solve(A, b)
