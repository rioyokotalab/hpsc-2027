// Replaces launch.cu of the ThunderKittens educational GEMM levels:
// times one level with CUDA events and checks it against cuBLAS.
#include "gemm_check.h"

int main(int argc, char **argv) {
  Problem p(argc > 1 ? atoi(argv[1]) : 4096);
  double t = p.ms([&] { matmul(p.A, p.B, p.C, p.N); });
  matmul(p.A, p.B, p.C, p.N);
  p.report(NAME, t);
}
