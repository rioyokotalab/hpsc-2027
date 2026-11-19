# A GPT-2-small-sized decoder (124M parameters) used by the Lecture 6 programs.
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PEAK = 989.4e12   # H100 SXM5 dense BF16 tensor-core peak, FLOP/s


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, vocab=50304, T=1024, d=768, L=12, h=12, ckpt=False):
        super().__init__()
        self.T, self.d, self.L, self.ckpt = T, d, L, ckpt
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, std=0.02)
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def forward(self, idx, tgt):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.ckpt else b(x)
        logits = self.head(self.ln(x))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))

    def flops_per_token(self):
        # PaLM (Chowdhery et al. 2022, App. B): 6N for the weights plus 12 L H Q T for attention (H Q = d)
        n = sum(p.numel() for p in self.parameters())
        return 6 * n + 12 * self.L * self.d * self.T
