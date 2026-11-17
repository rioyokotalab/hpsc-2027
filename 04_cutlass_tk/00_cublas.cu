// cuBLAS BF16 GEMM, the reference for every kernel in this folder.
#include "gemm_check.h"

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    Problem p(atoi(argv[i]));
    double t = p.ms([&] { p.cublas(); });
    cudaMemcpy(p.C, p.R, (size_t)p.N * p.N * 2, cudaMemcpyDeviceToDevice);
    p.report("cuBLAS", t);
  }
}
