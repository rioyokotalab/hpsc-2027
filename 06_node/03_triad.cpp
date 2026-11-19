#include <cstdio>
#include <omp.h>
// y = y + 3 x on 2^28 doubles (2 GiB per array): memory-bound, 24 bytes per element
int main() {
  const long n = 1L << 28;
  double *x = new double[n], *y = new double[n];
#pragma omp parallel for
  for (long i = 0; i < n; i++) { x[i] = 1; y[i] = 2; }  // first touch by the threads that use the data
  int maxt = omp_get_max_threads();
  const int threads[] = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192};
  for (int t : threads) {
    if (t > maxt) break;
    double best = 1e9;
    for (int r = 0; r < 3; r++) {
      double t0 = omp_get_wtime();
#pragma omp parallel for num_threads(t)
      for (long i = 0; i < n; i++) y[i] = y[i] + 3.0 * x[i];
      double dt = omp_get_wtime() - t0;
      if (dt < best) best = dt;
    }
    printf("threads %3d  %.4f s  %6.1f GB/s\n", t, best, 3.0 * 8 * n / best / 1e9);
  }
  delete[] x; delete[] y;
}
