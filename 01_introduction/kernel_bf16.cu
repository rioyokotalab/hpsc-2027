// The only file the agent may edit in the BF16 part.
// C = A * B for N x N row-major matrices; A, B are BF16, C is FP32 (FP32 accumulation).
// A, B, C are device pointers. run_gemm_bf16 is called by the harness; N is a multiple of 128.
#include <cuda_bf16.h>

__global__ void gemm_bf16(int N, const __nv_bfloat16 *A,
                          const __nv_bfloat16 *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
      sum += __bfloat162float(A[i*N + k]) * __bfloat162float(B[k*N + j]);
    C[i*N + j] = sum;
  }
}

void run_gemm_bf16(int N, const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C) {
  dim3 threads(32, 32);
  dim3 blocks((N + 31) / 32, (N + 31) / 32);
  gemm_bf16<<<blocks, threads>>>(N, A, B, C);
}
