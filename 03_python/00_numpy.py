import time
import numpy as np

N = 10_000_000
a = np.arange(N, dtype=np.float32) % 1000
b = np.ones(N, dtype=np.float32)

c = np.empty_like(a)
t = time.perf_counter()
for i in range(N):                         # one interpreted step per element
    c[i] = a[i] + b[i]
t_loop = time.perf_counter() - t

t = time.perf_counter()
d = a + b                                  # one call, compiled loop inside NumPy
t_numpy = time.perf_counter() - t

assert np.array_equal(c, d)
print(f"Python loop {t_loop*1e3:9.1f} ms  {t_loop/N*1e9:6.1f} ns/element")
print(f"NumPy       {t_numpy*1e3:9.1f} ms  {t_numpy/N*1e9:6.2f} ns/element")
