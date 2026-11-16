# Fixed harness file: do not edit.
# Usage: python3 plot_bf16.py [PEAK_TFLOPS]   (default 989.4, H100 SXM5 BF16 dense)
# Draws results_bf16.csv as steps_bf16.png: TFLOP/s per step, with cuBLAS BF16 and the BF16 peak.
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results_bf16.csv")))
peak = float(sys.argv[1]) if len(sys.argv) > 1 else 989.4
x = [int(r["step"]) for r in rows]
y = [float(r["tflops"]) for r in rows]
cublas = max(float(r["cublas_tflops"]) for r in rows)
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(x, y, "o-", color="#0072B2", lw=2)
for xi, yi in zip(x, y):
    ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
ax.axhline(cublas, color="#D55E00", ls="--", label=f"cuBLAS BF16 {cublas:.1f} TFLOP/s")
ax.axhline(peak, color="black", ls=":", label=f"BF16 dense peak {peak:g} TFLOP/s")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in rows], rotation=30, ha="right")
ax.set_ylabel("TFLOP/s (N = 4096)")
ax.set_title(f"BF16 GEMM steps on {rows[-1]['gpu']}")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("steps_bf16.png", dpi=150)
print("steps_bf16.png")
