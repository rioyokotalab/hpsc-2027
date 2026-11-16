#!/bin/bash
# Fixed harness file: do not edit.
# Builds kernel_bf16.cu with the BF16 harness, checks the rules, verifies C against cuBLAS BF16
# for N = 4096 and 2048 with fresh random inputs, and reports the speed at N = 4096.
# Usage: ./check_bf16.sh       (ARCH=sm_86 ./check_bf16.sh on a non-H100 GPU)
cd "$(dirname "$0")" || exit 1
BASE=${BASE:-origin/main}
FIXED="inputs.h harness_bf16.cu reference_bf16.cu Makefile check_bf16.sh step_bf16.sh profile_bf16.sh plot_bf16.py"
fail() { echo "verified no ($1)"; exit 1; }

# 1. Only kernel_bf16.cu may differ from the course repository
git diff --quiet "$BASE" -- $FIXED || fail "harness files differ from $BASE: $(git diff --name-only "$BASE" -- $FIXED | tr '\n' ' ')"

# 2. CUDA runtime only: no libraries, no files, no processes
bad=$(grep -nE '^[[:space:]]*#[[:space:]]*include' kernel_bf16.cu | grep -vE '<(cuda_runtime\.h|cuda\.h|cuda_bf16\.h|cuda_fp16\.h|mma\.h|cooperative_groups\.h|cuda/barrier|cuda/pipeline|cudaTypedefs\.h|cstdio|cstdint|cassert)>')
[ -z "$bad" ] || fail "include not allowed: $bad"
grep -qiE 'cublas|cutlass|cute::|cudnn|dlopen|fopen|system[[:space:]]*\(|popen|getenv' kernel_bf16.cu && fail "library, file or process use in kernel_bf16.cu"

# 3. Build; the timed program must not link any CUDA library besides the runtime
make -s ARCH=${ARCH:-sm_90} gemm_bf16 reference_bf16 >/dev/null || fail "build failed"
ldd ./gemm_bf16 | grep -qiE 'cublas|cudnn|cutlass' && fail "gemm_bf16 links a library"

# 4. Verify and time
tmp=$(mktemp "${TMPDIR:-/tmp}/gemm-C.XXXXXX")
status=0
for N in 2048 4096; do
  seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
  ./gemm_bf16 $N $seed "$tmp" || { status=1; break; }
  ./reference_bf16 $N $seed "$tmp" || status=1
done
rm -f "$tmp"
[ $status -eq 0 ] && echo "verified yes" || echo "verified no"
exit $status
