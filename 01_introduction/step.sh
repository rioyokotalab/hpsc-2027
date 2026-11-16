#!/bin/bash
# Fixed harness file: do not edit. Run by the teacher, not by the agent.
# Usage: ./step.sh "shared-memory tiling"
# Checks the current kernel.cu and, if verified, records it as the next step.
cd "$(dirname "$0")" || exit 1
label=${1:?usage: ./step.sh "technique"}
out=$(./check.sh 2>&1); status=$?
echo "$out"
[ $status -eq 0 ] || { echo "not recorded"; exit 1; }
[ -f results.csv ] || echo "step,label,gpu,N,tflops,cublas_tflops" > results.csv
step=$(( $(wc -l < results.csv) - 1 ))
gpu=$(echo "$out" | sed -n 's/^GPU \(.*\)  N 4096 .*/\1/p')
tf=$(echo "$out" | sed -n 's/^GPU .*  N 4096 .*  \([0-9.]*\) TFLOP\/s$/\1/p')
cb=$(echo "$out" | sed -n 's/^N 4096 .*cuBLAS \([0-9.]*\) TFLOP\/s$/\1/p')
echo "$step,\"$label\",\"$gpu\",4096,$tf,$cb" >> results.csv
git add kernel.cu results.csv
git commit -qm "GEMM step $step: $label ($tf TFLOP/s on $gpu)" -- kernel.cu results.csv
echo "recorded step $step: $tf TFLOP/s"
