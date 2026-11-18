# Lecture 5, Exercise 4: one tolerance at one size, or the error as K grows?
import importlib, torch
cand = importlib.import_module("07_candidates")

torch.manual_seed(0)
M = N = 1024
names = ("candidate_a", "candidate_b", "candidate_c")
print("1) torch.testing.assert_close against cuBLAS, K = 1024")
a = torch.randn(M, 1024, device="cuda").bfloat16()
b = torch.randn(1024, N, device="cuda").bfloat16()
for n in names:
    try:
        torch.testing.assert_close(getattr(cand, n)(a, b), a @ b)
        print(f"   {n}: passed")
    except AssertionError as e:
        why = next((l.strip() for l in str(e).splitlines() if l.strip().startswith("Mismatched")), "")
        print(f"   {n}: failed ({why})")

print("2) relative error against FP64 as K grows (BF16 output rounding alone: ~1.7e-3)")
print(f"   {'K':>6s} " + " ".join(f"{n:>12s}" for n in names))
for K in (256, 1024, 4096, 16384, 65536):
    a = torch.randn(M, K, device="cuda").bfloat16()
    b = torch.randn(K, N, device="cuda").bfloat16()
    R = a.double() @ b.double()
    errs = [((getattr(cand, n)(a, b).double() - R).norm() / R.norm()).item() for n in names]
    print(f"   {K:6d} " + " ".join(f"{e:12.2e}" for e in errs))
