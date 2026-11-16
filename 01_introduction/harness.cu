// Fixed harness file: do not edit.
// Usage: ./gemm N SEED OUT.bin
// Times run_gemm from kernel.cu with CUDA events and writes C for the reference check.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "inputs.h"

void run_gemm(int N, const float *A, const float *B, float *C);

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

int main(int argc, char **argv) {
  if (argc < 4) { printf("usage: %s N SEED OUT.bin\n", argv[0]); return 1; }
  const int N = atoi(argv[1]);
  const uint64_t seed = strtoull(argv[2], 0, 10);
  const size_t bytes = (size_t)N * N * sizeof(float);
  std::vector<float> A, B, C((size_t)N * N);
  make_inputs(N, seed, A, B);
  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dC, bytes));
  CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  // warm-up, also sets the number of timed calls (about 1 s in total, 3 to 20 calls)
  CHECK(cudaMemset(dC, 0xff, bytes));
  CHECK(cudaEventRecord(start));
  run_gemm(N, dA, dB, dC);
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
    run_gemm(N, dA, dB, dC);
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
