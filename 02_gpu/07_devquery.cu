#include <cstdio>

int main(void) {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("%s\n", p.name);
  printf("SMs %d\n", p.multiProcessorCount);
  printf("shared/SM %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
  printf("L2 %d MiB\n", p.l2CacheSize >> 20);
  printf("regs/SM %d\n", p.regsPerMultiprocessor);
  printf("threads/SM %d\n", p.maxThreadsPerMultiProcessor);
  printf("memory %zu MiB\n", p.totalGlobalMem >> 20);
}
