# hpsc-2027

High Performance Scientific Computing 2027. Sample code for each lecture is in a folder named `NN_topic`.

Student ID:

|            | Topic                                       | Sample code |
| ---------- | ------------------------------------------- | ----------- |
| Lecture 1  | Why HPC limits AI                           | 01_introduction |
| Lecture 2  | GPU architecture and roofline               | 02_gpu      |
| Lecture 3  | Kernels I: Triton / CUDA-Python             | 03_python   |
| Lecture 4  | Kernels II: CUTLASS / ThunderKittens        | 04_cutlass_tk |
| Lecture 5  | Precision and numerics                      | 05_precision |
| Lecture 6  | Node-level optimization and parallelism     | 06_node     |
| Lecture 7  | Distributed training strategies             | 07_distributed |
| Lecture 8  | Interconnect and scaling measurement        |             |
| Lecture 9  | Data pipeline and I/O                       |             |
| Lecture 10 | Inference and serving                       |             |
| Lecture 11 | RL / agent training compute                 |             |
| Lecture 12 | AI for Science                              |             |
| Lecture 13 | Failure diagnosis and reproducibility       |             |
| Lecture 14 | Autonomous experiment loops and resources   |             |

## Submitting homework

Every homework is submitted by pushing to your own fork of this repository. Nothing else needs to be sent.

1. Fork `rioyokotalab/hpsc-2027` on GitHub, once, before Lecture 2.
2. In your fork, write your student ID after `Student ID:` at the top of this README, so that I know who you are.
3. Add the SSH public key of your TSUBAME4.0 account to GitHub, then clone your fork there and add this repository as `upstream`:

```
git clone git@github.com:<your-github-account>/hpsc-2027.git
cd hpsc-2027
git remote add upstream https://github.com/rioyokotalab/hpsc-2027.git
```

4. Before each lecture, get the new sample code: `git pull upstream main`.
5. Do the homework in that lecture's folder: change the homework program there, and write `homework.md` in the same folder with your prediction, the measured results, your diagnosis, your decision and the commands you ran. Then commit and push:

```
git add 02_gpu/17_bucket_sort.cu 02_gpu/homework.md
git commit -m "Lecture 2 homework"
git push origin main
```

| Lecture | Homework program in your fork |
| ------- | ----------------------------- |
| 2 | `02_gpu/17_bucket_sort.cu` |
| 3 | `03_python/10_rmsnorm.py` |
| 4 | `04_cutlass_tk/30_linear_relu.cu` |
| 5 | `05_precision/10_mxp.py` |
| 6 | `06_node/my_train.py` (start from a copy of `30_train.py`) |
| 7 | `07_distributed/my_train.py` (start from a copy of `50_train.py`) |

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
