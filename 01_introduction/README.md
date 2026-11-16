# 01_introduction: GEMM with a coding agent

A coding agent makes a matrix multiplication (C = A B, N x N, row-major FP32) faster,
one technique per step. The agent may edit only `kernel.cu`. Everything else is a fixed
harness:

| File | Role |
| --- | --- |
| `kernel.cu` | the kernel; step 0 is the naive one-thread-per-element version |
| `harness.cu` | random inputs from a seed, C poisoned with NaN before every call, timing with CUDA events |
| `reference.cu` | a separate cuBLAS program (FP32, no TF32) that recomputes C from the same seed and checks every element |
| `check.sh` | rules (harness unchanged, CUDA runtime only), build, verify at N = 2048 and 4096, speed at N = 4096 |
| `profile.sh` | short Nsight Compute summary for the next prompt |
| `step.sh` | teacher only: check again and commit the step to `results.csv` |
| `plot.py` | `steps.png` from `results.csv` |
| `prompt.md` | the prompt for every step |

```
qrsh -g <group> -l gpu_1=1 -l h_rt=1:00:00
module load cuda
./check.sh                      # ARCH=sm_86 ./check.sh on other GPUs
./profile.sh
```

An element is correct when |C - C_cuBLAS| <= 1e-4 (|A| |B|)_ij, that is, relative to
the size of the terms it sums. Inputs change on every run, so a kernel cannot store answers.

## Steps

FP32 without tensor cores (Boehm, A6000, % of cuBLAS): naive 1.3, global-memory
coalescing 8.5, shared-memory caching 12.8, 1-D block tiling 36.5, 2-D block tiling
68.7, vectorized loads 78.4, warp tiling 93.7
(<https://siboehm.com/articles/22/CUDA-MMM>).

BF16 with Hopper tensor cores (H100 SXM, 4096^3, cuBLAS 716 TFLOP/s): tensor-core-free
BF16 32, WGMMA + TMA 317, larger tiles 423, producer/consumer warp specialization
498, two consumer groups 610-631, L2-aware tile order 660, 764 after 11 steps
(<https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog>).
The BF16 part needs a BF16 harness, which is not here yet.
