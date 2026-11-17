// Homework: y = relu(x W + b), the linear layer of an MLP, in ONE kernel.
// x is M x K, W is K x N, b has N entries; all BF16, row-major.
// This version uses two kernels: a CUTLASS GEMM, then bias_relu.
// Fuse the bias and the ReLU into the CUTLASS epilogue (see the fusion
// operations in cutlass/epilogue/fusion/operations.hpp), keep the check.
// Usage: ./30_linear_relu [M N K]          (default 8192 8192 1024)
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace cute;

using T = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    T, RowMajor, 8, T, RowMajor, 8, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, T, RowMajor, 8, T, RowMajor, 8, float,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

__global__ void bias_relu(T *y, const T *b, int M, int N) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < (size_t)M * N) y[i] = T(fmaxf(float(y[i]) + float(b[i % N]), 0.0f));
}

struct Linear {                             // everything your version needs
  Gemm gemm;
  typename Gemm::Arguments args;
  void *work;
  T *x, *W, *b, *y;
  int M, N, K;
  Linear(T *x_, T *W_, T *b_, T *y_, int M_, int N_, int K_) : x(x_), W(W_), b(b_), y(y_), M(M_), N(N_), K(K_) {
    auto sx = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {M, K, 1});
    auto sW = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {N, K, 1});
    auto sy = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {M, N, 1});
    args = {cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1}, {x, sx, W, sW}, {{1.0f, 0.0f}, y, sy, y, sy}};
    cudaMalloc(&work, Gemm::get_workspace_size(args) + 1);
    gemm.initialize(args, work);
  }
  void run() {                              // y = relu(x W + b)
    gemm.run();
    bias_relu<<<((size_t)M * N + 255) / 256, 256>>>(y, b, M, N);
  }
};

int main(int argc, char **argv) {
  int M = argc > 3 ? atoi(argv[1]) : 8192, N = argc > 3 ? atoi(argv[2]) : 8192, K = argc > 3 ? atoi(argv[3]) : 1024;
  std::vector<T> hx((size_t)M * K), hW((size_t)K * N), hb(N);
  srand(1);
  for (auto &v : hx) v = T(rand() / (float)RAND_MAX - 0.5f);
  for (auto &v : hW) v = T(rand() / (float)RAND_MAX - 0.5f);
  for (auto &v : hb) v = T(rand() / (float)RAND_MAX - 0.5f);
  T *x, *W, *b, *y, *r;
  cudaMalloc(&x, hx.size() * 2); cudaMalloc(&W, hW.size() * 2); cudaMalloc(&b, N * 2);
  cudaMalloc(&y, (size_t)M * N * 2); cudaMalloc(&r, (size_t)M * N * 2);
  cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(W, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(b, hb.data(), N * 2, cudaMemcpyHostToDevice);

  // reference: cuBLAS GEMM, then bias and ReLU in FP32 on the CPU
  cublasHandle_t h; cublasCreate(&h);
  float one = 1, zero = 0;
  float *rf; cudaMalloc(&rf, (size_t)M * N * 4);
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, N, x, CUDA_R_16BF, K,
               &zero, rf, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  std::vector<float> ref((size_t)M * N);
  cudaMemcpy(ref.data(), rf, ref.size() * 4, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < ref.size(); i++) ref[i] = fmaxf(ref[i] + float(hb[i % N]), 0.0f);

  Linear lin(x, W, b, y, M, N, K);
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  lin.run(); lin.run();                     // warm-up
  cudaEventRecord(t0);
  for (int r = 0; r < 20; r++) lin.run();
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 20;

  std::vector<T> hy((size_t)M * N);
  cudaMemcpy(hy.data(), y, hy.size() * 2, cudaMemcpyDeviceToHost);
  double e = 0, m = 0;
  for (size_t i = 0; i < hy.size(); i++) { e = fmax(e, fabs(float(hy[i]) - ref[i])); m = fmax(m, fabs(ref[i])); }
  cudaError_t err = cudaGetLastError();
  printf("M %d N %d K %d  %.3f ms  %.1f TFLOP/s  error %.1e  %s\n", M, N, K, ms, 2.0 * M * N * K / ms / 1e9, e / m,
         err != cudaSuccess ? cudaGetErrorString(err) : e / m < 1e-2 ? "ok" : "FAIL");
}
