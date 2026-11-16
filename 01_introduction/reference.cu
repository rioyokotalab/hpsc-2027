// Fixed harness file: do not edit.
// Usage: ./reference N SEED C.bin
// Recomputes C = A * B with cuBLAS (FP32, no TF32) from the same seed and compares.
// Error of each element is scaled by (|A| |B|)_ij, the size of the terms it sums.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include "inputs.h"

int main(int argc, char **argv) {
  if (argc < 4) { printf("usage: %s N SEED C.bin\n", argv[0]); return 1; }
  const int N = atoi(argv[1]);
  const uint64_t seed = strtoull(argv[2], 0, 10);
  const size_t n2 = (size_t)N * N, bytes = n2 * sizeof(float);
  std::vector<float> A, B, C(n2), R(n2), S(n2);
  make_inputs(N, seed, A, B);
  FILE *f = fopen(argv[3], "rb");
  if (!f || fread(C.data(), sizeof(float), n2, f) != n2) { printf("verified no (cannot read %s)\n", argv[3]); return 1; }
  fclose(f);
  float *dA, *dB, *dR;
  cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dR, bytes);
  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
  const float one = 1, zero = 0;
  // row-major C = A B is column-major C^T = B^T A^T
  auto gemm = [&](const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &out) {
    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, N, dA, N, &zero, dR, N);
    cudaMemcpy(out.data(), dR, bytes, cudaMemcpyDeviceToHost);
  };
  gemm(A, B, R);
  // cuBLAS speed on the same inputs, for the plot
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, N, dA, N, &zero, dR, N);
  cudaEventRecord(start);
  for (int r = 0; r < 10; r++)
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, N, dA, N, &zero, dR, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  double t = ms / 1e3 / 10;
  for (auto &a : A) a = fabsf(a);
  for (auto &b : B) b = fabsf(b);
  gemm(A, B, S);
  double worst = 0;
  size_t bad = 0;
  for (size_t i = 0; i < n2; i++) {
    double e = std::isfinite(C[i]) ? fabs((double)C[i] - R[i]) / S[i] : INFINITY;
    if (!(e <= 1e-4)) bad++;
    if (!(e <= worst)) worst = e;
  }
  printf("N %d  verified %s  wrong %zu  max scaled error %.2e  cuBLAS %.2f TFLOP/s\n",
         N, bad ? "no" : "yes", bad, worst, 2.0 * N * N * N / t / 1e12);
  return bad ? 1 : 0;
}
