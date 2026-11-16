#include <cstdio>

__global__ void add(const float *a, const float *b, float *c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x)
    c[i] = a[i] + b[i];
}

int main(void) {
  const size_t N = 1 << 28;
  const int M = 256;
  const int R = 10;
  float *a, *b, *c;
  cudaMallocManaged(&a, N*sizeof(float));
  cudaMallocManaged(&b, N*sizeof(float));
  cudaMallocManaged(&c, N*sizeof(float));
  for (size_t i=0; i<N; i++) {
    a[i] = i % 1000;
    b[i] = 1;
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int B=1; B<=1024; B*=2) {
    cudaMemset(c, 0, N*sizeof(float));
    add<<<B,M>>>(a, b, c, N); // warm-up
    cudaDeviceSynchronize();
    for (size_t i=0; i<N; i++) {
      if (c[i] != a[i] + b[i]) {
        printf("B %d: error at %zu\n", B, i);
        return 1;
      }
    }
    add<<<B,M>>>(a, b, c, N); // warm-up after the CPU check
    cudaEventRecord(start);
    for (int r=0; r<R; r++)
      add<<<B,M>>>(a, b, c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double t = ms / 1e3 / R;
    printf("blocks %4d  threads %6d  %.1f GB/s\n", B, B*M, 12.0 * N / t / 1e9);
  }
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
