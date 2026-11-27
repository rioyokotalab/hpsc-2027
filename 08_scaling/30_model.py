# Homework template: a performance model of DDP training on 1 to 8 GPUs (and beyond).
# No GPU needed. Fit the parameters below to job_output_example.txt, then run
#   python check_model.py 30_model.py
# All times in seconds, sizes in bytes.

GPUS_PER_NODE = 4


def t_allreduce(S, p):
    """Time of one all-reduce of S bytes on p GPUs (p <= 4: one node, else p/4 nodes).
    Fit latency and bandwidth per case from the 10_allreduce sections, not from the ping-pong."""
    if p == 1:
        return 0.0
    latency = 30e-6               # TODO: fit
    bandwidth = 100e9             # TODO: bus bandwidth, fit (one node? two nodes? which message size?)
    return latency + 2 * (p - 1) / p * S / bandwidth


def t_compute(B):
    """Forward + backward + optimizer of the 1.5B GPT with B sequences per GPU, no communication."""
    return 0.1 + 0.02 * B         # TODO: fit to the no_sync times


def t_step(p, B, params=1.56e9, bucket=25e6):
    """DDP step: compute plus the exposed part of the gradient all-reduce (FP32, 25 MB buckets)."""
    buckets = 4 * params / bucket
    return t_compute(B) + buckets * t_allreduce(bucket, p)   # TODO: how much of this does DDP hide?
