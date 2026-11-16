# hpsc-2027

High Performance Scientific Computing 2027. Sample code for each lecture is in a folder named `NN_topic`.

Student ID:

|            | Topic                                       | Sample code |
| ---------- | ------------------------------------------- | ----------- |
| Lecture 1  | Why HPC limits AI                           | 01_introduction |
| Lecture 2  | GPU architecture and roofline               | 02_gpu      |
| Lecture 3  | Kernels I: Triton / CUDA-Python             |             |
| Lecture 4  | Kernels II: CUTLASS / ThunderKittens        |             |
| Lecture 5  | Precision and numerics                      |             |
| Lecture 6  | Node-level optimization and parallelism     |             |
| Lecture 7  | Distributed training strategies             |             |
| Lecture 8  | Interconnect and scaling measurement        |             |
| Lecture 9  | Data pipeline and I/O                       |             |
| Lecture 10 | Inference and serving                       |             |
| Lecture 11 | RL / agent training compute                 |             |
| Lecture 12 | AI for Science                              |             |
| Lecture 13 | Failure diagnosis and reproducibility       |             |
| Lecture 14 | Autonomous experiment loops and resources   |             |

## Building on TSUBAME4.0

```
qrsh -g <group> -l gpu_1=1 -l h_rt=0:30:00
module load cuda
cd 02_gpu
make 00_hello && ./a.out
```

The Makefiles target the H100 (`-arch=sm_90`); on another GPU use `make ARCH=sm_86 ...` or its own architecture.
