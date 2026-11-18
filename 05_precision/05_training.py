# Lecture 5: one small training run in five precisions.
# A 3-layer MLP learns a fixed random "teacher" MLP; same data, same seed, Adam.
import sys, torch

torch.manual_seed(0)
D, H, NS, STEPS = 512, 2048, 8192, 1500
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-5
mlp = lambda *d: torch.nn.Sequential(*[m for i in range(len(d) - 1) for m in
                                       (torch.nn.Linear(d[i], d[i + 1]), torch.nn.GELU())][:-1]).cuda()
teacher = mlp(D, H, D)
X = torch.randn(NS, D, device="cuda")
with torch.no_grad():
    Y = teacher(X)

def train(mode):
    torch.manual_seed(1)
    model = mlp(D, H, H, D)
    if mode == "BF16 weights, no master":
        model = model.bfloat16()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler() if "scaler" in mode else None
    amp = torch.bfloat16 if mode.startswith("BF16 autocast") else torch.float16
    g = torch.Generator(device="cuda").manual_seed(2)
    losses = []
    for step in range(STEPS + 1):
        i = torch.randint(0, NS, (256,), device="cuda", generator=g)
        x = X[i].bfloat16() if mode == "BF16 weights, no master" else X[i]
        with torch.autocast("cuda", dtype=amp, enabled="autocast" in mode):
            loss = torch.nn.functional.mse_loss(model(x).float(), Y[i])
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if step % 100 == 0:
            losses.append(loss.item())
    return losses

print(f"Adam, lr = {LR}, loss every 100 steps")
for mode in ("FP32", "BF16 autocast (FP32 master)", "BF16 weights, no master", "FP16 autocast", "FP16 autocast + scaler"):
    print(f"{mode:30s} " + " ".join(f"{l:.4f}" for l in train(mode)))

# How many gradient values would FP16 flush to zero? A deeper network, one backward pass.
torch.manual_seed(0)
deep = mlp(*([512] * 9), 1)
x, y = torch.randn(256, 512, device="cuda"), torch.randn(256, 1, device="cuda")
torch.nn.functional.mse_loss(deep(x), y).backward()
g = torch.cat([p.grad.flatten() for p in deep.parameters()]).abs()
h = torch.histc(g[g > 0].log2(), bins=24, min=-48, max=0).long().tolist()
print("\nlog2 |gradient| histogram, bins of width 2 from -48 to 0:", " ".join(map(str, h)))
print(f"9-layer MLP, FP32 gradients: {(g < 2**-25).float().mean().item():.1%} below 2^-25 (FP16 rounds them to 0), "
      f"{(g < 2**-14).float().mean().item():.1%} below 2^-14 (FP16 smallest normal)")
