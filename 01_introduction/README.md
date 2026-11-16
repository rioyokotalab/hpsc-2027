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

## BF16 with tensor cores

The same loop with BF16 inputs: A and B are BF16, C is FP32 and every sum is accumulated
in FP32. The agent may edit only `kernel_bf16.cu`, whose entry is
`run_gemm_bf16(int N, const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C)`.

| File | Role |
| --- | --- |
| `kernel_bf16.cu` | the kernel; step 0 is the naive one-thread-per-element version without tensor cores |
| `harness_bf16.cu` | the inputs of `inputs.h` rounded to BF16 (`__float2bfloat16_rn`), C poisoned with NaN before every call, timing with CUDA events |
| `reference_bf16.cu` | a separate cuBLAS program (`cublasGemmEx`, BF16 inputs, `CUBLAS_COMPUTE_32F`, FP32 C) that rounds the inputs the same way, checks every element and reports the cuBLAS BF16 speed |
| `check_bf16.sh` | rules (BF16 harness unchanged, CUDA runtime only), build, verify at N = 2048 and 4096, speed at N = 4096 |
| `profile_bf16.sh` | short Nsight Compute summary, including how busy the tensor pipe is |
| `step_bf16.sh` | teacher only: check again and commit the step to `results_bf16.csv` |
| `plot_bf16.py` | `steps_bf16.png` from `results_bf16.csv`, with the H100 SXM BF16 dense peak of 989.4 TFLOP/s |
| `prompt_bf16.md` | the prompt for every step |

```
module load cuda
./check_bf16.sh
./profile_bf16.sh
```

The tolerance is the same, |C - C_cuBLAS| <= 1e-4 (|A| |B|)_ij. Both programs round the
same FP32 numbers to BF16, and the product of two BF16 numbers (8-bit significands) is
exact in FP32 (24 bits), so the kernel and cuBLAS add exactly the same terms and differ
only in the order and rounding of the FP32 additions. That difference is of order
sqrt(N) 2^-24, about 4e-6 at N = 4096, far below 1e-4. A kernel that drops a single
term is off by about |a b| / (|A| |B|)_ij, of order 1/N (2.4e-4 at N = 4096, 5e-4 at
N = 2048), so it fails; so does one that accumulates in BF16 (error of order 2^-9).

Step 1 on an H100 (one thread per element, then WMMA 16 x 16 x 16 BF16 fragments with
128 x 128 shared-memory tiles) is in `results_bf16.csv` after `./step_bf16.sh`.
