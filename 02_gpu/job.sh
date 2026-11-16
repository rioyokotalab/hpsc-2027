#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:05:00
# Usage: qsub -g <group> job.sh              runs Exercises 1-3
#        qsub -g <group> job.sh ./00_hello   runs one program
# Build first on the login node: module load cuda; make
module load cuda
if [ $# -gt 0 ]; then
  "$@"
else
  ./07_devquery
  for e in 10 12 14 16 18 20 22 24 26 28; do ./vecadd $((1 << e)); done
  ./09_threads
fi
