# hpsc-2027

High Performance Scientific Computing 2027. Sample code for each lecture is in a folder named `NN_topic`.

Student ID:

|            | Topic                                       | Sample code |
| ---------- | ------------------------------------------- | ----------- |
| Lecture 1  | Why HPC limits AI                           | 01_introduction |
| Lecture 2  | GPU architecture and roofline               | 02_gpu      |
| Lecture 3  | Kernels I: Triton / CUDA-Python             | 03_python   |
| Lecture 4  | Kernels II: CUTLASS / ThunderKittens        | 04_cutlass_tk |
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

Build on the login node and run on a GPU node with a short batch job:

```
cd 02_gpu
module load cuda
make                                  # every CUDA program, nvcc -arch=sm_90
qsub -g <group> job.sh                # Exercises 1-3 on one H100
qsub -g <group> job.sh ./00_hello     # or one program
```

OpenACC programs: `module purge; module load nvhpc; make openacc` (the nvhpc and cuda modules conflict).
The Makefiles target the H100 (`-arch=sm_90`, `-gpu=cc90`); on another GPU use `make ARCH=sm_86 ...` or its own architecture.
