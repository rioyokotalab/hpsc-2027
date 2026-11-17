#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:08:00
# Usage: qsub -g <group> job.sh                       runs Exercise 1 (the staircase)
#        qsub -g <group> job.sh ./10_cutlass_gemm 8192  runs one program
# Build first on the login node (see README.md): module load cuda; make
module load cuda
if [ $# -gt 0 ]; then
  "$@"
else
  ./00_cublas 4096 8192
  for L in 02 03 04 05 06 07 08; do ./level_$L 4096; done
  ./tk_h100 | grep -E "time|TFLOP"
  ./10_cutlass_gemm 4096 8192
fi
