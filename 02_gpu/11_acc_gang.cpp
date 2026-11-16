#include <cstdio>
#include <openacc.h>

int main() {
#pragma acc parallel loop num_gangs(2) num_workers(2) vector_length(32)
  for (int i=0; i<128; i++) {
    if (i % 16 == 0)
      printf("i %3d  gang %d  worker %d  vector %2d\n",
             i, __pgi_gangidx(), __pgi_workeridx(), __pgi_vectoridx());
  }
}
