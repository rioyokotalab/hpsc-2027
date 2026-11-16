// The only file the agent may edit.
// C = A * B for N x N row-major FP32 matrices; A, B, C are device pointers.
// run_gemm is called by the harness; N is a multiple of 128.

__global__ void gemm(int N, const float *A,
                     const float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
      sum += A[i*N + k] * B[k*N + j];
    C[i*N + j] = sum;
  }
}

void run_gemm(int N, const float *A, const float *B, float *C) {
  dim3 threads(32, 32);
  dim3 blocks((N + 31) / 32, (N + 31) / 32);
  gemm<<<blocks, threads>>>(N, A, B, C);
}
