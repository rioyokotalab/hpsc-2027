// Inputs, cuBLAS reference, error and timing for row-major BF16 C = A * B (N x N).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>

struct Problem {
  int N;
  __nv_bfloat16 *A, *B, *C, *R;            // R: cuBLAS result
  cublasHandle_t h;
  explicit Problem(int n) : N(n) {
    size_t NN = (size_t)N * N;
    std::vector<__nv_bfloat16> hA(NN), hB(NN);
    srand(1);
    for (size_t i = 0; i < NN; i++) hA[i] = __float2bfloat16(rand() / (float)RAND_MAX - 0.5f);
    for (size_t i = 0; i < NN; i++) hB[i] = __float2bfloat16(rand() / (float)RAND_MAX - 0.5f);
    cudaMalloc(&A, NN * 2); cudaMalloc(&B, NN * 2); cudaMalloc(&C, NN * 2); cudaMalloc(&R, NN * 2);
    cudaMemcpy(A, hA.data(), NN * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), NN * 2, cudaMemcpyHostToDevice);
    cudaMemset(C, 0, NN * 2);
    cublasCreate(&h);
    cublas();
  }
  void cublas() {                          // row-major C = A B is column-major C^T = B^T A^T
    float one = 1, zero = 0;
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, B, CUDA_R_16BF, N, A, CUDA_R_16BF, N,
                 &zero, R, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  double error() {                         // max |C - R| / max |R|
    size_t NN = (size_t)N * N;
    std::vector<__nv_bfloat16> hC(NN), hR(NN);
    cudaMemcpy(hC.data(), C, NN * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hR.data(), R, NN * 2, cudaMemcpyDeviceToHost);
    double e = 0, m = 0;
    for (size_t i = 0; i < NN; i++) {
      double c = __bfloat162float(hC[i]), r = __bfloat162float(hR[i]);
      e = fmax(e, fabs(c - r)); m = fmax(m, fabs(r));
    }
    return e / m;
  }
  template <class F> double ms(F f) {      // average time of one call with CUDA events
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    f(); f();                              // warm-up
    cudaEventRecord(t0); f(); cudaEventRecord(t1); cudaEventSynchronize(t1);
    float one; cudaEventElapsedTime(&one, t0, t1);
    int reps = one > 100 ? 1 : (one > 5 ? 5 : 20);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; r++) f();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float t; cudaEventElapsedTime(&t, t0, t1);
    return t / reps;
  }
  void report(const char *name, double t) {
    double e = error();
    cudaError_t err = cudaGetLastError();
    printf("%-22s N %5d  %9.3f ms  %6.1f TFLOP/s  error %.1e  %s\n", name, N, t,
           2.0 * N * N * N / t / 1e9, e, err != cudaSuccess ? cudaGetErrorString(err) : e < 1e-2 ? "ok" : "FAIL");
  }
};
