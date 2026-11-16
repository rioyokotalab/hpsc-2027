# Prompt for one BF16 optimization step

Fill in the angle brackets from the previous step (`./check_bf16.sh`, `./profile_bf16.sh`).

```
Context   kernel_bf16.cu (current version)
          last run: N = 4096, <TFLOP/s>, verified = yes
          Nsight Compute: <DRAM throughput>, <SM busy>,
          <tensor pipe busy>, <occupancy>

Task      Apply ONE technique: <e.g. WMMA tensor-core fragments>.

Before    Explain why it should be faster and predict
coding    the speed-up (x) from the numbers above.

Rules     Edit kernel_bf16.cu only. No cuBLAS, CUTLASS or CuTe.
          Do not touch the harness or the reference.
          A, B are BF16; accumulate and write C in FP32.
          N is a multiple of 128.

After     Run ./check_bf16.sh. Report: verified, TFLOP/s,
coding    predicted vs measured, what limits it now.
```
