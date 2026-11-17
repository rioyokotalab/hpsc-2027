# Lecture 4: Tensor-core kernels (ThunderKittens, CUTLASS)

BF16 matrix multiplication on one H100, from a simple loop to cuBLAS speed.

## Setup (once, on the TSUBAME4.0 login node)

```
cd 04_cutlass_tk
git clone https://github.com/HazyResearch/ThunderKittens.git thunderkittens
git -C thunderkittens checkout 67845f5fa48d05343dbbc1ba1403f11061f08d2d
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git
module load cuda
make -j 8                                  # under a minute
```

## Run

```
qsub -g <group> job.sh                     # Exercise 1: the staircase, about 30 s
qsub -g <group> job.sh ./10_cutlass_gemm 4096 8192
qsub -g <group> job.sh ./level_08 4000     # outside the contract (Exercise 4)
qsub -g <group> job.sh ./30_linear_relu    # homework
```

Change the CUTLASS configuration at build time, for example
`make -B 10_cutlass_gemm TILE="-DTM=128 -DTN=128 -DTK=64 -DCM=1 -DSTAGES=3"`.
Schedules: `-DSCHEDULE=cutlass::gemm::KernelTmaWarpSpecializedCooperative`
(with `-DEPILOGUE=cutlass::epilogue::TmaWarpSpecializedCooperative`) or
`-DSCHEDULE=cutlass::gemm::KernelTmaWarpSpecializedPingpong`
(with `-DEPILOGUE=cutlass::epilogue::TmaWarpSpecialized`).

`20_attention.py` runs with the Lecture 3 Python environment:
`qsub -g <group> ../03_python/job.sh python ../04_cutlass_tk/20_attention.py` (about 100 s).

| File | What it shows |
| --- | --- |
| `gemm_check.h` | inputs, cuBLAS reference, error, timing with CUDA events |
| `00_cublas.cu` | cuBLAS BF16 GEMM |
| `launch.cu` | times and checks one ThunderKittens educational level (`level_02` … `level_08`) |
| `tk_h100` | the full ThunderKittens H100 GEMM, with its own timing |
| `10_cutlass_gemm.cu` | a GEMM from CUTLASS collectives; tile, cluster, stages, schedule |
| `20_attention.py` | attention: three PyTorch operations against fused kernels |
| `30_linear_relu.cu` | homework: `relu(x W + b)` in one kernel |

All matrices are N × N, BF16, row-major, with FP32 accumulation. The
ThunderKittens educational levels assume N is a multiple of 64.
