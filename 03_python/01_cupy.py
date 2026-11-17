import time
import cupy as cp

N = 1 << 28
a = cp.arange(N, dtype=cp.float32) % 1000
b = cp.ones(N, dtype=cp.float32)
c = a + b                                  # warm-up

t = time.perf_counter()
c = a + b                                  # returns when the kernel is queued
t_wrong = time.perf_counter() - t

start, stop = cp.cuda.Event(), cp.cuda.Event()
start.record()
c = a + b
stop.record()
stop.synchronize()                         # wait until the GPU is done
t_right = cp.cuda.get_elapsed_time(start, stop) / 1e3

assert bool((c == a + 1).all())
print(f"perf_counter without sync {t_wrong*1e3:7.3f} ms  {12*N/t_wrong/1e9:9.1f} GB/s")
print(f"CUDA events               {t_right*1e3:7.3f} ms  {12*N/t_right/1e9:9.1f} GB/s")
