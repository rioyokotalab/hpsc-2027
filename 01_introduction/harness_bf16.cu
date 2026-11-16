// Fixed harness file: do not edit.
// Usage: ./gemm_bf16 N SEED OUT.bin
// Times run_gemm_bf16 from kernel_bf16.cu with CUDA events and writes C (FP32) for the reference check.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_bf16.h>
#include "inputs.h"

void run_gemm_bf16(int N, const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C);

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

int main(int argc, char **argv) {
  if (argc < 4) { printf("usage: %s N SEED OUT.bin\n", argv[0]); return 1; }
  const int N = atoi(argv[1]);
  const uint64_t seed = strtoull(argv[2], 0, 10);
  const size_t n2 = (size_t)N * N, bytes16 = n2 * sizeof(__nv_bfloat16), bytes = n2 * sizeof(float);
  std::vector<float> A, B, C(n2);
  make_inputs(N, seed, A, B);
  // BF16 inputs: round the FP32 inputs to nearest, exactly as reference_bf16.cu does
  std::vector<__nv_bfloat16> A16(n2), B16(n2);
  for (size_t i = 0; i < n2; i++) A16[i] = __float2bfloat16_rn(A[i]);
  for (size_t i = 0; i < n2; i++) B16[i] = __float2bfloat16_rn(B[i]);
  __nv_bfloat16 *dA, *dB;
  float *dC;
  CHECK(cudaMalloc(&dA, bytes16));
  CHECK(cudaMalloc(&dB, bytes16));
  CHECK(cudaMalloc(&dC, bytes));
  CHECK(cudaMemcpy(dA, A16.data(), bytes16, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, B16.data(), bytes16, cudaMemcpyHostToDevice));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  // warm-up, also sets the number of timed calls (about 1 s in total, 3 to 20 calls)
  CHECK(cudaMemset(dC, 0xff, bytes));
  CHECK(cudaEventRecord(start));
  run_gemm_bf16(N, dA, dB, dC);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaGetLastError());
  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  int R = ms > 0 ? (int)(1000 / ms) : 20;
  R = R < 3 ? 3 : (R > 20 ? 20 : R);
  // C is poisoned (NaN) before every timed call, so each call must compute all of C
  double total = 0;
  for (int r = 0; r < R; r++) {
    CHECK(cudaMemset(dC, 0xff, bytes));
    CHECK(cudaEventRecord(start));
    run_gemm_bf16(N, dA, dB, dC);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    total += ms;
  }
  double t = total / 1e3 / R;
  CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
  FILE *f = fopen(argv[3], "wb");
  fwrite(C.data(), sizeof(float), C.size(), f);
  fclose(f);
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  printf("GPU %s  N %d  calls %d  time %.3f ms  %.2f TFLOP/s\n", p.name, N, R, t * 1e3, 2.0 * N * N * N / t / 1e12);
}
