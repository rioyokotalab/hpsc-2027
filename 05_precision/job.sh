#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:08:00
# Usage: qsub -g <group> job.sh                         runs Exercises 1-3 and the demos
#        qsub -g <group> job.sh python check_mxp.py     runs one program
# Uses the Python environment of Lecture 3 (see ../03_python/README.md).
module load python/3.14.10
. $HOME/hpsc-venv/bin/activate
export TRITON_DEFAULT_FP_FUSION=1
if [ $# -gt 0 ]; then
  "$@"
else
  for f in 00_formats 01_gemm 02_sum 03_accumulate 04_scaling 05_training 06_reproducible 08_refine; do
    echo "=== $f"
    python $f.py
  done
fi
