# Prompt for one optimization step

Fill in the angle brackets from the previous step (`./check.sh`, `./profile.sh`).

```
Context   kernel.cu (current version)
          last run: N = 4096, <TFLOP/s>, verified = yes
          Nsight Compute: <DRAM throughput>, <SM busy>, <occupancy>

Task      Apply ONE technique: <e.g. shared-memory tiling>.

Before    Explain why it should be faster and predict
coding    the speed-up (x) from the numbers above.

Rules     Edit kernel.cu only. No cuBLAS, CUTLASS or CuTe.
          Do not touch the harness or the reference.
          N is a multiple of 128.

After     Run ./check.sh. Report: verified, TFLOP/s,
coding    predicted vs measured, what limits it now.
```
