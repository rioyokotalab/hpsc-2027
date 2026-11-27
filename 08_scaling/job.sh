#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=0:15:00
# Usage: qsub -g <group> job.sh     two full nodes, 8 H100s; runs Exercises 1-3 (about 10 min)
# Lecture 8 runs this job once for the whole class; its output is job_output_example.txt.
# Uses the Python environment of Lecture 3 (see ../03_python/README.md).
module load python/3.14.10 openmpi/5.0.10-gcc
. $HOME/hpsc-venv/bin/activate
export MASTER_ADDR=$(head -1 $PE_HOSTFILE | cut -d' ' -f1) MASTER_PORT=29500
awk '{print $1" slots=4"}' $PE_HOSTFILE > hosts.$JOB_ID
M="mpirun --hostfile hosts.$JOB_ID -x MASTER_ADDR -x MASTER_PORT -x PATH -x LD_LIBRARY_PATH -x VIRTUAL_ENV"
echo "=== nodes"; cut -d' ' -f1 $PE_HOSTFILE; nvidia-smi topo -m | head -9
echo "=== 00_pingpong";  $M -npernode 4 -n 8 python 00_pingpong.py
echo "=== 10_allreduce"; $M -npernode 4 -n 8 python 10_allreduce.py
echo "# NCCL_ALGO=Ring";       $M -npernode 4 -n 8 -x NCCL_ALGO=Ring python 10_allreduce.py all
echo "# NCCL_ALGO=Tree";       $M -npernode 4 -n 8 -x NCCL_ALGO=Tree python 10_allreduce.py all
echo "# NCCL_IB_HCA==mlx5_1 (one NIC)"; $M -npernode 4 -n 8 -x NCCL_IB_HCA==mlx5_1 python 10_allreduce.py all
echo "=== 20_scaling"
for np in 1 2 4 8; do
  $M -npernode $(( np < 4 ? np : 4 )) -n $np python 20_scaling.py 1600 48 25 1 2 4 8    # 1.5B GPT
done
for np in 4 8; do $M -npernode 4 -n $np python 20_scaling.py 1024 24 16 16; done          # 355M GPT
rm -f hosts.$JOB_ID
