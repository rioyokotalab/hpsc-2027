#include <cstdio>
#include <chrono>

void step(float *x, int N) {
#pragma acc parallel loop present(x[0:N])
  for (int i=0; i<N; i++)
    x[i] = 0.5f * x[i] + 1.0f;
}

void step_copy(float *x, int N) {
#pragma acc parallel loop copy(x[0:N])
  for (int i=0; i<N; i++)
    x[i] = 0.5f * x[i] + 1.0f;
}

int main() {
  const int N = 1 << 26, T = 100;
  float *x = new float[N];
  for (int i=0; i<N; i++) x[i] = 0;
  step_copy(x, N); // warm-up
  auto t0 = std::chrono::steady_clock::now();
  for (int t=0; t<T; t++)
    step_copy(x, N);  // copies x to the GPU and back every step
  auto t1 = std::chrono::steady_clock::now();
#pragma acc data copy(x[0:N])
  {
    for (int t=0; t<T; t++)
      step(x, N);     // x stays on the GPU
  }
  auto t2 = std::chrono::steady_clock::now();
  printf("copy every step: %.1f ms\n", std::chrono::duration<double>(t1 - t0).count() * 1e3);
  printf("one data region: %.1f ms\n", std::chrono::duration<double>(t2 - t1).count() * 1e3);
  printf("x[0] = %g (expected 2)\n", x[0]);
  delete[] x;
}
