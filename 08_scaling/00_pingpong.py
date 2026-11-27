# Ping-pong: one-way time and bandwidth of one message, inside a node (rank 0 <-> 1)
# and between nodes (rank 0 <-> 4), with PyTorch send/recv over NCCL (Exercise 1).
# Each pair gets its own two-GPU group: in the eight-GPU group NCCL splits its
# point-to-point channels among all peers and one pair sees far less bandwidth.
import time, torch, torch.distributed as dist, dist_env
r, p = dist_env.init()
for peer, where in ((1, "inside a node"), (4, "between nodes")):
    g = dist.new_group([0, peer])                 # every rank must call new_group
    if r == 0: print(f"# rank 0 <-> rank {peer}: {where}", flush=True)
    for n in [2**k for k in range(2, 31, 2)]:
        x = torch.ones(max(n // 2, 1), device="cuda", dtype=torch.bfloat16)
        reps = 200 if n < 2**20 else 20
        for it in range(2):                       # first pass warms up
            torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
            for _ in range(reps if it else 10):
                if r == 0: dist.send(x, peer, group=g); dist.recv(x, peer, group=g)
                elif r == peer: dist.recv(x, 0, group=g); dist.send(x, 0, group=g)
            torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / reps / 2
        if r == 0: print(f"pingpong {where:13s} {n:11d} bytes one-way {t*1e6:9.2f} us {n/t/1e9:7.2f} GB/s", flush=True)
dist.destroy_process_group()
