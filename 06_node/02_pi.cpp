#include <cstdio>
#include <omp.h>
// pi = integral of 4/(1+x^2) from 0 to 1: compute-bound, 2 x 10^9 points
int main() {
  const long n = 2000000000;
  const double dx = 1. / n;
  int maxt = omp_get_max_threads();
  const int threads[] = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192};
  for (int t : threads) {
    if (t > maxt) break;
    double pi = 0, t0 = omp_get_wtime();
#pragma omp parallel for reduction(+:pi) num_threads(t)
    for (long i = 0; i < n; i++) {
      double x = (i + 0.5) * dx;
      pi += 4.0 / (1.0 + x * x) * dx;
    }
    printf("threads %3d  pi = %.15f  %.4f s\n", t, pi, omp_get_wtime() - t0);
  }
}
