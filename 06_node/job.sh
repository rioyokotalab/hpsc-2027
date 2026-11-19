#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:10:00
# Usage: qsub -g <group> job.sh                              runs Exercises 1-4 and the demos (about 5 min)
#        qsub -g <group> job.sh python check_train.py my_train.py   runs one command
# One full node: 192 CPU cores and 4 H100s. Build the C++ programs first with make on the login node.
# Uses the Python environment of Lecture 3 (see ../03_python/README.md).
module load python/3.14.10
. $HOME/hpsc-venv/bin/activate
export TRITON_DEFAULT_FP_FUSION=1
# Thread placement for the OpenMP programs only: exported, it would also pin every
# torchrun rank's main thread to the same first core.
OMP="env OMP_PLACES=cores OMP_PROC_BIND=spread"
if [ $# -gt 0 ]; then
  "$@"
else
  nvidia-smi topo -m
  echo "=== 00_hello";  $OMP OMP_NUM_THREADS=4 ./00_hello
  echo "=== 01_race";   $OMP OMP_NUM_THREADS=192 ./01_race
  echo "=== 02_pi";     $OMP OMP_NUM_THREADS=192 ./02_pi
  echo "=== 03_triad";  $OMP OMP_NUM_THREADS=192 ./03_triad
  echo "=== 04_copy";   python 04_copy.py
  echo "=== 10_step";   python 10_step.py
  echo "=== 20_allreduce"; torchrun --standalone --nproc_per_node=4 20_allreduce.py
  echo "=== 21_ddp";    torchrun --standalone --nproc_per_node=4 21_ddp.py
fi
