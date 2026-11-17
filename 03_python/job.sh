#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:08:00
# Usage: qsub -g <group> job.sh                     runs Exercises 1-3
#        qsub -g <group> job.sh python check_silu_mul.py   runs one program
# Set up the environment once on the login node (see README.md).
module load python/3.14.10
. $HOME/hpsc-venv/bin/activate
export TRITON_DEFAULT_FP_FUSION=1        # keep today's default, silence the warning
if [ $# -gt 0 ]; then
  "$@"
else
  python 01_cupy.py
  python 02_numba.py
  python 03_triton_add.py
  python 04_softmax.py
  python 05_matmul.py
fi
