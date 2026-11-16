#!/bin/bash
# Fixed harness file: do not edit.
# Builds kernel.cu with the harness, checks the rules, verifies C against cuBLAS
# for N = 4096 and 2048 with fresh random inputs, and reports the speed at N = 4096.
# Usage: ./check.sh            (ARCH=sm_86 ./check.sh on a non-H100 GPU)
cd "$(dirname "$0")" || exit 1
BASE=${BASE:-origin/main}
FIXED="inputs.h harness.cu reference.cu Makefile check.sh step.sh profile.sh plot.py"
fail() { echo "verified no ($1)"; exit 1; }

# 1. Only kernel.cu may differ from the course repository
git diff --quiet "$BASE" -- $FIXED || fail "harness files differ from $BASE: $(git diff --name-only "$BASE" -- $FIXED | tr '\n' ' ')"

# 2. CUDA runtime only: no libraries, no files, no processes
bad=$(grep -nE '^[[:space:]]*#[[:space:]]*include' kernel.cu | grep -vE '<(cuda_runtime\.h|cuda\.h|cuda_bf16\.h|cuda_fp16\.h|mma\.h|cooperative_groups\.h|cuda/barrier|cuda/pipeline|cudaTypedefs\.h|cstdio|cstdint|cassert)>')
[ -z "$bad" ] || fail "include not allowed: $bad"
grep -qiE 'cublas|cutlass|cute::|cudnn|dlopen|fopen|system[[:space:]]*\(|popen|getenv' kernel.cu && fail "library, file or process use in kernel.cu"

# 3. Build; the timed program must not link any CUDA library besides the runtime
make -s ARCH=${ARCH:-sm_90} gemm reference >/dev/null || fail "build failed"
ldd ./gemm | grep -qiE 'cublas|cudnn|cutlass' && fail "gemm links a library"

# 4. Verify and time
tmp=$(mktemp "${TMPDIR:-/tmp}/gemm-C.XXXXXX")
status=0
for N in 2048 4096; do
  seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
  ./gemm $N $seed "$tmp" || { status=1; break; }
  ./reference $N $seed "$tmp" || status=1
done
rm -f "$tmp"
[ $status -eq 0 ] && echo "verified yes" || echo "verified no"
exit $status
