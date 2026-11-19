#include <cstdio>
#include <cstdlib>
#include <omp.h>
// A histogram of 10^8 random keys in 16 bins, three ways
int main() {
  const long n = 100000000;
  const int range = 16;
  int *key = new int[n];
  for (long i = 0; i < n; i++) key[i] = rand() % range;
  long count[range];
  for (int k = 0; k < 3; k++) {
    for (int j = 0; j < range; j++) count[j] = 0;
    double t0 = omp_get_wtime();
    if (k == 0) {
#pragma omp parallel for
      for (long i = 0; i < n; i++) count[key[i]]++;        // race: updates are lost
    } else if (k == 1) {
#pragma omp parallel for
      for (long i = 0; i < n; i++) {
#pragma omp atomic
        count[key[i]]++;                                     // one update at a time
      }
    } else {
#pragma omp parallel for reduction(+:count[:range])
      for (long i = 0; i < n; i++) count[key[i]]++;        // private copies, added at the end
    }
    double dt = omp_get_wtime() - t0;
    long sum = 0;
    for (int j = 0; j < range; j++) sum += count[j];
    const char *name[] = {"race", "atomic", "reduction"};
    printf("%-9s total = %9ld (expected %ld)  %.4f s\n", name[k], sum, n, dt);
  }
  delete[] key;
}
