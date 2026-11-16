#include <cstdio>

int main() {
  const int N = 1 << 24;
  int sum = 0;
#pragma acc parallel loop reduction(+:sum)
  for (int i=0; i<N; i++)
    sum += 1;
  printf("%d (expected %d)\n", sum, N);
}
