# Homework checker: runs a training program on the 4 GPUs of one node and compares its
# losses with the losses of the template 50_train.py.
# Usage: python check_train.py my_train.py
#        python check_train.py --ref            (writes ref_loss.txt from 50_train.py)
import re, subprocess, sys, time

REF = "ref_loss.txt"
TOL = 0.001                   # relative difference allowed per printed step (BF16, non-deterministic sums)

prog = "50_train.py" if sys.argv[1] == "--ref" else sys.argv[1]
cmd = ["torchrun", "--standalone", "--nproc_per_node=4", prog]
t0 = time.perf_counter()
out = subprocess.run(cmd, capture_output=True, text=True)
wall = time.perf_counter() - t0
if out.returncode != 0:
    print(out.stdout[-2000:], out.stderr[-4000:])
    sys.exit(f"FAIL: {prog} exited with code {out.returncode}")
loss = {int(s): float(l) for s, l in re.findall(r"^step\s+(\d+) loss ([\d.]+)", out.stdout, re.M)}
m = re.search(r"^train time ([\d.]+) s", out.stdout, re.M)
mem = re.search(r"^peak memory ([\d.]+) GB", out.stdout, re.M)
if sys.argv[1] == "--ref":
    open(REF, "w").write("".join(f"{s} {l}\n" for s, l in sorted(loss.items())))
    print(f"wrote {REF}: {len(loss)} losses, train time {m.group(1)} s, end-to-end {wall:.1f} s")
    sys.exit()
ref = {int(s): float(l) for s, l in (line.split() for line in open(REF))}
worst = 0.0
for s, r in ref.items():
    if s not in loss:
        sys.exit(f"FAIL: no loss printed for step {s}")
    d = abs(loss[s] - r) / r
    worst = max(worst, d)
    print(f"step {s:4d}  loss {loss[s]:.4f}  reference {r:.4f}  difference {100 * d:5.2f} %")
if not m:
    sys.exit("FAIL: no 'train time' line")
train = float(m.group(1))
print(f"largest loss difference {100 * worst:.2f} % (allowed {100 * TOL:.1f} %)")
print(f"train time {train:.2f} s, end-to-end {wall:.1f} s (includes Python, CUDA and NCCL start-up)")
if mem:
    print(f"peak memory {mem.group(1)} GB on GPU 0")
print("PASS" if worst <= TOL and train < wall else "FAIL")
