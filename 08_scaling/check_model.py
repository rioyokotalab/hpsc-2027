# Homework checker: compares t_step(p, B) of your model with the DDP steps measured
# in job_output_example.txt (1.5B GPT, p = 1, 2, 4, 8 GPUs, B = 1, 2, 4, 8 sequences per GPU).
# Usage: python check_model.py 30_model.py
import importlib.util, re, sys
TOL = 0.15                                        # relative error allowed per point
spec = importlib.util.spec_from_file_location("model", sys.argv[1]); model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
pat = r"^scaling p=(\d+) params (\d+) M batch (\d+)/GPU .*?step\s+([\d.]+) ms"
meas = {(int(p), int(b)): float(t) / 1e3 for p, M, b, t in re.findall(pat, open("job_output_example.txt").read(), re.M) if int(M) > 1000}
if not meas:
    sys.exit("FAIL: no 1.5B scaling lines in job_output_example.txt")
worst, bad = 0.0, 0
print(f"{'p':>3} {'B':>3} {'measured ms':>12} {'model ms':>9} {'error':>7}")
for (p, B), t in sorted(meas.items()):
    pred = model.t_step(p, B); e = (pred - t) / t; worst = max(worst, abs(e)); bad += abs(e) > TOL
    print(f"{p:3d} {B:3d} {t*1e3:12.1f} {pred*1e3:9.1f} {100*e:+6.1f}%{'  <-' if abs(e) > TOL else ''}")
print(f"{'PASS' if bad == 0 else 'FAIL'}: {len(meas) - bad} of {len(meas)} points within {TOL:.0%}; worst {worst:.1%}")
for p in (16, 64):
    try: print(f"model, p = {p:2d}, B = 8: step {model.t_step(p, 8)*1e3:7.1f} ms")
    except Exception as ex: print(f"model, p = {p:2d}: {type(ex).__name__}: {ex}")
