import torch

def bench(fn, reps=20):
    """Average time of one call in ms, measured with CUDA events."""
    fn()                                   # warm-up (and JIT compile)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / reps
