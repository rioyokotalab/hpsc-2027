#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 1 << 20;
  float *a = new float[N], *b = new float[N], *c = new float[N];
  for (int i=0; i<N; i++) {
    a[i] = i % 1000;
    b[i] = 1;
  }
#pragma acc parallel loop copyin(a[0:N], b[0:N]) copyout(c[0:N])
  for (int i=0; i<N; i++)
    c[i] = a[i] + b[i];
  for (int i=0; i<N; i++)
    if (c[i] != a[i] + b[i]) { printf("error at %d\n", i); return 1; }
  printf("N %d ok\n", N);
  delete[] a; delete[] b; delete[] c;
}
