// Fixed harness file: do not edit.
// Deterministic random inputs in [-1, 1) from a seed (splitmix64).
#pragma once
#include <cstdint>
#include <vector>

static inline uint64_t splitmix64(uint64_t &x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static inline void make_inputs(int N, uint64_t seed, std::vector<float> &A, std::vector<float> &B) {
  A.resize((size_t)N * N);
  B.resize((size_t)N * N);
  uint64_t s = seed;
  for (auto &a : A) a = (splitmix64(s) >> 40) * (2.0f / (1 << 24)) - 1.0f;
  for (auto &b : B) b = (splitmix64(s) >> 40) * (2.0f / (1 << 24)) - 1.0f;
}
