#!/bin/bash
# Fixed harness file: do not edit. Run by the teacher, not by the agent.
# Usage: ./step_bf16.sh "WMMA tensor cores"
# Checks the current kernel_bf16.cu and, if verified, records it as the next BF16 step.
cd "$(dirname "$0")" || exit 1
label=${1:?usage: ./step_bf16.sh "technique"}
out=$(./check_bf16.sh 2>&1); status=$?
echo "$out"
[ $status -eq 0 ] || { echo "not recorded"; exit 1; }
[ -f results_bf16.csv ] || echo "step,label,gpu,N,tflops,cublas_tflops" > results_bf16.csv
step=$(( $(wc -l < results_bf16.csv) - 1 ))
gpu=$(echo "$out" | sed -n 's/^GPU \(.*\)  N 4096 .*/\1/p')
tf=$(echo "$out" | sed -n 's/^GPU .*  N 4096 .*  \([0-9.]*\) TFLOP\/s$/\1/p')
cb=$(echo "$out" | sed -n 's/^N 4096 .*cuBLAS BF16 \([0-9.]*\) TFLOP\/s$/\1/p')
echo "$step,\"$label\",\"$gpu\",4096,$tf,$cb" >> results_bf16.csv
git add kernel_bf16.cu results_bf16.csv
git commit -qm "BF16 GEMM step $step: $label ($tf TFLOP/s on $gpu)" -- kernel_bf16.cu results_bf16.csv
echo "recorded step $step: $tf TFLOP/s"
