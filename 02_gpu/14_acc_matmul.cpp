#include <cstdio>
#include <cstdlib>
#include <chrono>

void matmul(const float *A, const float *B, float *C, int N) {
#pragma acc parallel loop gang present(A[0:N*N], B[0:N*N], C[0:N*N])
  for (int i=0; i<N; i++) {
#pragma acc loop vector
    for (int j=0; j<N; j++) {
      float sum = 0;
      for (int k=0; k<N; k++)
        sum += A[N*i+k] * B[N*k+j];
      C[N*i+j] = sum;
    }
  }
}

int main(int argc, char **argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 4096;
  float *A = new float[N*N], *B = new float[N*N], *C = new float[N*N];
  for (int i=0; i<N*N; i++) {
    A[i] = drand48() - 0.5;
    B[i] = drand48() - 0.5;
  }
#pragma acc data copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
  {
    matmul(A, B, C, N); // warm-up
    auto tic = std::chrono::steady_clock::now();
    matmul(A, B, C, N);
    auto toc = std::chrono::steady_clock::now();
    double t = std::chrono::duration<double>(toc - tic).count();
    printf("N %d  %.3f s  %.2f TFLOP/s\n", N, t, 2.0 * N * N * N / t / 1e12);
  }
  // spot check 16 elements against the CPU
  double err = 0;
  for (int s=0; s<16; s++) {
    int i = (s * 7919) % N, j = (s * 104729) % N;
    double c = 0;
    for (int k=0; k<N; k++) c += (double)A[N*i+k] * B[N*k+j];
    double e = c - C[N*i+j]; if (e < 0) e = -e; if (e > err) err = e;
  }
  printf("max error of 16 checked elements %.2e\n", err);
  delete[] A; delete[] B; delete[] C;
}
