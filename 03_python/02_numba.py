import time
import numpy as np
from numba import cuda

@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)                       # blockIdx.x * blockDim.x + threadIdx.x
    if i < c.size:
        c[i] = a[i] + b[i]

N = 1 << 28
M = 256
a = cuda.to_device(np.arange(N, dtype=np.float32) % 1000)
b = cuda.to_device(np.ones(N, dtype=np.float32))
c = cuda.device_array(N, dtype=np.float32)
blocks = (N + M - 1) // M

t = time.perf_counter()
add[blocks, M](a, b, c)                    # first call compiles the kernel
cuda.synchronize()
t_first = time.perf_counter() - t

start, stop = cuda.event(), cuda.event()
start.record()
add[blocks, M](a, b, c)
stop.record()
stop.synchronize()
t = cuda.event_elapsed_time(start, stop) / 1e3

assert np.array_equal(c.copy_to_host(), a.copy_to_host() + 1)
print(f"first call (compile + run) {t_first*1e3:8.1f} ms")
print(f"second call                {t*1e3:8.3f} ms  {12*N/t/1e9:7.1f} GB/s")
