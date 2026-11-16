#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:10:00
module load cuda
./vecadd 268435456
