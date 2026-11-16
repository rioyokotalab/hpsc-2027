#include <cstdio>
#include <cstdlib>

__global__ void twice(float *a, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  a[i] *= 2;
}

int main(void) {
  const int N = 2000;
  const int M = 1024;
  const size_t bytes = N * sizeof(float);
  // Explicit copies: one array on the CPU, one on the GPU
  float *h = (float*)malloc(bytes);
  for (int i=0; i<N; i++) h[i] = i;
  float *d;
  cudaMalloc(&d, bytes);
  cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
  twice<<<(N+M-1)/M,M>>>(d, N);
  cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
  printf("explicit: a[%d] = %g\n", N-1, h[N-1]);
  cudaFree(d);
  free(h);
  // Managed memory: one pointer for the CPU and the GPU
  float *a;
  cudaMallocManaged(&a, bytes);
  for (int i=0; i<N; i++) a[i] = i;
  twice<<<(N+M-1)/M,M>>>(a, N);
  cudaDeviceSynchronize();
  printf("managed:  a[%d] = %g\n", N-1, a[N-1]);
  cudaFree(a);
}
