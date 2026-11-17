import importlib
import torch
add_mod = importlib.import_module("03_triton_add")

a = torch.ones(1 << 20, device="cuda")
k = add_mod.add_kernel[(1024,)](a, a, a, a.numel(), BLOCK=1024)
print("stages:", list(k.asm.keys()))
print("registers", k.n_regs, " warps", k.metadata.num_warps)
for line in k.asm["ptx"].splitlines():
    if "ld.global" in line or "st.global" in line:
        print(line.strip())
