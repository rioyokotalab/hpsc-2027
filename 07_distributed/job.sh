#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:15:00
# Usage: qsub -g <group> job.sh                                   runs Exercises 1-4 and the demo (about 10 min)
#        qsub -g <group> job.sh python check_train.py my_train.py  runs one command
# One full node: 4 H100s joined by NVLink. Uses the Python environment of Lecture 3 (see ../03_python/README.md).
module load python/3.14.10
. $HOME/hpsc-venv/bin/activate
TR="torchrun --standalone --nproc_per_node=4"
if [ $# -gt 0 ]; then
  "$@"
else
  echo "=== 00_collectives";     $TR 00_collectives.py
  echo "=== 10_memory"
  for m in ddp zero1 fsdp fsdp_noreshard; do $TR 10_memory.py $m 1600 48 25 4; done
  for m in ddp fsdp; do $TR 10_memory.py $m 4096 32 32 1; done
  echo "=== 20_tensor_parallel";  $TR 20_tensor_parallel.py
  echo "=== 30_pipeline";         $TR 30_pipeline.py
  echo "=== 40_ring_attention";   $TR 40_ring_attention.py
fi
