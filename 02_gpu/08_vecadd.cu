#include <cstdio>
#include <cstdlib>

__global__ void add(const float *a, const float *b, float *c, size_t N) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= N) return;
  c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
  const size_t N = argc > 1 ? atol(argv[1]) : 1 << 28;
  const int M = 256;
  const int R = 10;
  const size_t blocks = (N + M - 1) / M;
  float *a, *b, *c;
  cudaMallocManaged(&a, N*sizeof(float));
  cudaMallocManaged(&b, N*sizeof(float));
  cudaMallocManaged(&c, N*sizeof(float));
  for (size_t i=0; i<N; i++) {
    a[i] = i % 1000;
    b[i] = 1;
  }
  add<<<blocks,M>>>(a, b, c, N); // warm-up; also moves the pages to the GPU
  cudaDeviceSynchronize();
  for (size_t i=0; i<N; i++) {
    if (c[i] != a[i] + b[i]) {
      printf("error at %zu: %g\n", i, c[i]);
      return 1;
    }
  }
  add<<<blocks,M>>>(a, b, c, N); // warm-up after the CPU check
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int r=0; r<R; r++)
    add<<<blocks,M>>>(a, b, c, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  double t = ms / 1e3 / R;
  printf("N %zu  time %.3f ms  %.1f GB/s\n", N, t * 1e3, 12.0 * N / t / 1e9);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
