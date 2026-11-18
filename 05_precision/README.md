# Lecture 5: Precision and numerics

How many bits each value gets, what that costs in accuracy and buys in speed, measured on one H100.
Everything uses PyTorch and Triton from the Lecture 3 environment (`~/hpsc-venv`, see `../03_python/README.md`).

## Run

```
cd 05_precision
qsub -g <group> job.sh                              # Exercises 1-3 and every demo (about 3 min)
qsub -g <group> job.sh python check_candidates.py   # Exercise 4
qsub -g <group> job.sh python check_mxp.py          # Homework
```

| File | What it shows |
| --- | --- |
| `00_formats.py` | unit roundoff and range of each format, one rounding, a counter that stops (Exercise 1) |
| `01_gemm.py` | one matrix multiplication in FP64, FP32, TF32, BF16, FP16 and FP8: speed and error |
| `02_sum.py` | the error of a sum in three orders |
| `03_accumulate.py` | error against the inner dimension K for each accumulator (Exercise 2) |
| `kernels.py` | a Triton matmul whose accumulator is FP32 or BF16 |
| `04_scaling.py` | scales and outliers in INT8, FP8 and FP4; FP8 GEMM speed (Exercise 3) |
| `05_training.py` | one small training run in five precisions; gradients below FP16's range |
| `06_reproducible.py` | atomics and run-to-run differences; batch invariance |
| `07_candidates.py`, `check_candidates.py` | Exercise 4: three kernels an agent might hand you |
| `08_refine.py` | iterative refinement: FP32 LU with FP64 residuals |
| `10_mxp.py`, `check_mxp.py` | Homework: HPL-MxP on one GPU |
| `bench.py` | timing with CUDA events |
