// Fixed harness file: do not edit.
// Usage: ./reference_bf16 N SEED C.bin
// Recomputes C = A * B with cuBLAS (BF16 inputs, FP32 compute and output) from the same seed
// and compares. Error of each element is scaled by (|A| |B|)_ij, the size of the terms it sums.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "inputs.h"

int main(int argc, char **argv) {
  if (argc < 4) { printf("usage: %s N SEED C.bin\n", argv[0]); return 1; }
  const int N = atoi(argv[1]);
  const uint64_t seed = strtoull(argv[2], 0, 10);
  const size_t n2 = (size_t)N * N, bytes16 = n2 * sizeof(__nv_bfloat16), bytes = n2 * sizeof(float);
  std::vector<float> A, B, C(n2), R(n2), S(n2);
  make_inputs(N, seed, A, B);
  // the same rounding as harness_bf16.cu
  std::vector<__nv_bfloat16> A16(n2), B16(n2);
  for (size_t i = 0; i < n2; i++) A16[i] = __float2bfloat16_rn(A[i]);
  for (size_t i = 0; i < n2; i++) B16[i] = __float2bfloat16_rn(B[i]);
  FILE *f = fopen(argv[3], "rb");
  if (!f || fread(C.data(), sizeof(float), n2, f) != n2) { printf("verified no (cannot read %s)\n", argv[3]); return 1; }
  fclose(f);
  __nv_bfloat16 *dA, *dB;
  float *dR;
  cudaMalloc(&dA, bytes16); cudaMalloc(&dB, bytes16); cudaMalloc(&dR, bytes);
  cublasHandle_t h;
  cublasCreate(&h);
  const float one = 1, zero = 0;
  // row-major C = A B is column-major C^T = B^T A^T
  auto run = [&]() {
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, N,
                 &zero, dR, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  };
  auto gemm = [&](const std::vector<__nv_bfloat16> &a, const std::vector<__nv_bfloat16> &b, std::vector<float> &out) {
    cudaMemcpy(dA, a.data(), bytes16, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes16, cudaMemcpyHostToDevice);
    run();
    cudaMemcpy(out.data(), dR, bytes, cudaMemcpyDeviceToHost);
  };
  gemm(A16, B16, R);
  // cuBLAS BF16 tensor-core speed on the same inputs, for the plot
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  run();
  cudaEventRecord(start);
  for (int r = 0; r < 10; r++) run();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  double t = ms / 1e3 / 10;
  // |x| of a BF16 number is exact in BF16
  for (auto &a : A16) a = __float2bfloat16_rn(fabsf(__bfloat162float(a)));
  for (auto &b : B16) b = __float2bfloat16_rn(fabsf(__bfloat162float(b)));
  gemm(A16, B16, S);
  double worst = 0;
  size_t bad = 0;
  for (size_t i = 0; i < n2; i++) {
    double e = std::isfinite(C[i]) ? fabs((double)C[i] - R[i]) / S[i] : INFINITY;
    if (!(e <= 1e-4)) bad++;
    if (!(e <= worst)) worst = e;
  }
  printf("N %d  verified %s  wrong %zu  max scaled error %.2e  cuBLAS BF16 %.2f TFLOP/s\n",
         N, bad ? "no" : "yes", bad, worst, 2.0 * N * N * N / t / 1e12);
  return bad ? 1 : 0;
}
