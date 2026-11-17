# Lecture 3: GPU kernels in Python

CuPy, Numba CUDA and Triton on one H100. Everything runs in one batch job.

## Setup (once, on the TSUBAME4.0 login node, about 3 minutes, 6 GB in your home)

```
module load python/3.14.10
python3 -m venv ~/hpsc-venv
. ~/hpsc-venv/bin/activate
pip install torch "numba-cuda[cu13]" cupy-cuda13x "numpy<2.5" matplotlib
```

`numpy<2.5`: NumPy 2.5 removed `np.row_stack`, which numba-cuda 0.30 still uses.

## Run

```
cd 03_python
qsub -g <group> job.sh python 00_numpy.py           # CPU only, but time it on a compute node:
                                                     # on the shared login node it was 100x slower
qsub -g <group> job.sh                               # Exercises 1-3 on one H100 (about 3 min)
qsub -g <group> job.sh python check_silu_mul.py      # Exercise 4
qsub -g <group> job.sh ./profile.sh python 09_profile.py   # Nsight Compute
```

| File | What it shows |
| --- | --- |
| `00_numpy.py` | a Python loop against one NumPy call |
| `01_cupy.py` | CuPy, and why `perf_counter` without synchronisation is wrong |
| `02_numba.py` | a CUDA kernel in Python with Numba; first call compiles |
| `03_triton_add.py` | vector addition in Triton against PyTorch, N = 2^10 to 2^28 |
| `04_softmax.py` | five PyTorch operations against one fused Triton kernel |
| `05_matmul.py` | a tiled Triton matmul against cuBLAS (FP32, TF32, BF16) |
| `06_tiles.py` | tile size, registers, spills and speed |
| `07_inspect.py` | the PTX Triton generated for vector addition |
| `08_silu_mul.py`, `check_silu_mul.py` | Exercise 4: a kernel written by your agent, and its check |
| `09_profile.py`, `profile.sh` | Nsight Compute metrics for four kernels |
| `bench.py` | timing with CUDA events |
