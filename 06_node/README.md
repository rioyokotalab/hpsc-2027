# Lecture 6: Node-level optimization and parallelism

One TSUBAME4.0 node: two 96-core CPUs and four H100s joined by NVLink.
Threads on the CPU cores with OpenMP, one training step of a 124M-parameter GPT on one GPU
(eager, activation checkpointing, `torch.compile`), and data-parallel training on the four GPUs with NCCL.
The Python programs use the environment of Lecture 3 (`~/hpsc-venv`, see `../03_python/README.md`).

## Run

```
cd 06_node
make                                              # the OpenMP programs, on the login node
qsub -g <group> job.sh                            # Exercises 1-4 and every demo (one node, about 5 min)
qsub -g <group> job.sh torchrun --standalone --nproc_per_node=4 21_ddp.py compile   # optional: compiled DDP
qsub -g <group> job.sh python check_train.py my_train.py                           # Homework
```

| File | What it shows |
| --- | --- |
| `00_hello.cpp` | an OpenMP parallel region |
| `01_race.cpp` | a histogram with a race, with atomic, and with a reduction (Exercise 1) |
| `02_pi.cpp` | π by integration on 1 to 192 threads: compute-bound (Exercise 1) |
| `03_triad.cpp` | y = y + 3x on 1 to 192 threads: memory-bound (Exercise 1) |
| `04_copy.py` | copies between GPUs (NVLink) and from the host (PCIe) |
| `gpt.py` | the 124M GPT and its FLOPs per token |
| `10_step.py` | one training step on one H100: eager, checkpointing, compiled; time, MFU, memory, kernels (Exercise 2) |
| `20_allreduce.py` | NCCL all-reduce on 4 GPUs, 4 KiB to 1 GiB (Exercise 3) |
| `21_ddp.py` | DistributedDataParallel on 4 GPUs: no all-reduce, 25 MB buckets, one bucket (Exercise 4) |
| `30_train.py`, `check_train.py` | Homework: one training run on one node |
