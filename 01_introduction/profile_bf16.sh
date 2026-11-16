#!/bin/bash
# Fixed harness file: do not edit.
# Short Nsight Compute summary of one BF16 GEMM call at N = 4096, for the next prompt.
cd "$(dirname "$0")" || exit 1
make -s ARCH=${ARCH:-sm_90} gemm_bf16 >/dev/null || exit 1
tmp=$(mktemp "${TMPDIR:-/tmp}/gemm-C.XXXXXX")
out=$(ncu --launch-skip 1 --launch-count 1 --csv --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread \
  ./gemm_bf16 4096 1 "$tmp" 2>&1)
rm -f "$tmp"
echo "$out" | python3 -c '
import csv, sys
lines = [l for l in sys.stdin if l.startswith("\"")]
names = {"dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM throughput",
         "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM busy",
         "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": "tensor pipe busy",
         "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy",
         "launch__registers_per_thread": "registers per thread"}
found = 0
for r in csv.DictReader(lines):
    n = r.get("Metric Name")
    if n in names:
        found += 1
        print(names[n] + ": " + r["Metric Value"] + " " + r["Metric Unit"])
sys.exit(0 if found else 1)
' || { echo "profiling failed:"; echo "$out" | grep -E '==ERROR==' | head -3; exit 1; }
