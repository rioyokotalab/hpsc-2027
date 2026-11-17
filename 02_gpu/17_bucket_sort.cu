#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Homework: execute the bucket sort on the GPU.
// Usage: ./17_bucket_sort [n] [range]      (default n = 50, range = 5)
int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 50;
  int range = argc > 2 ? atoi(argv[2]) : 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++)
    key[i] = rand() % range;
  if (n <= 50) {
    for (int i=0; i<n; i++) printf("%d ",key[i]);
    printf("\n");
  }
  std::vector<int> count(range, 0);        // for the check at the end
  for (int i=0; i<n; i++) count[key[i]]++;

  auto tic = std::chrono::steady_clock::now();
  // Execute this part on the GPU
  std::vector<int> bucket(range);
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  auto toc = std::chrono::steady_clock::now();

  if (n <= 50) {
    for (int i=0; i<n; i++) printf("%d ",key[i]);
    printf("\n");
  }
  bool ok = true;                          // sorted, and every value kept its count
  for (int i=1; i<n; i++) ok &= key[i-1] <= key[i];
  for (int i=0; i<n; i++) count[key[i]]--;
  for (int i=0; i<range; i++) ok &= count[i] == 0;
  double ms = std::chrono::duration<double, std::milli>(toc - tic).count();
  printf("n %d  range %d  %.3f ms  %s\n", n, range, ms, ok ? "ok" : "FAIL");
}
