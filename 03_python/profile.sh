#!/bin/sh
# Usage (inside a GPU job): ./profile.sh python 09_profile.py
# Prints time, DRAM traffic, memory and compute throughput, occupancy, registers per kernel.
module load cuda                           # provides ncu (Nsight Compute)
ncu --profile-from-start off --csv --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread "$@" |
python -c '
import csv, sys
rows = [r for r in csv.reader(sys.stdin) if len(r) > 10]
head, rows = rows[0], rows[1:]
k, m, v, u = (head.index(c) for c in ("Kernel Name", "Metric Name", "Metric Value", "Metric Unit"))
order, val = [], {}
for r in rows:
    key = (r[0], r[k][:40])
    if key not in val: order.append(key); val[key] = {}
    val[key][r[m]] = (r[v], r[u])
short = {"gpu__time_duration.sum": "time", "dram__bytes_read.sum": "DRAM read",
         "dram__bytes_write.sum": "DRAM write",
         "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory %",
         "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute %",
         "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy %",
         "launch__registers_per_thread": "registers"}
for key in order:
    print(key[1])
    print("   " + "  ".join(f"{short[n]} {val[key][n][0]} {val[key][n][1]}".strip()
                            for n in short if n in val[key]))
'
