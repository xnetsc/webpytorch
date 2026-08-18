"""#4 proof: idiomatic pytorch GPT code (nanoGPT-style) runs UNMODIFIED through
the torchshim (backed by webtorch) — including training. Both backends."""
import json, math
import numpy as np
from js import pythonIO
import cupy as cp
backend = cp.get_backend_name()
print(f"backend={backend}")

from webtorch import torchshim
torchshim.install()          # register torch / torch.nn / torch.nn.functional / torch.optim

# ===== BELOW: written as ordinary pytorch. Not adapted for webtorch. =====
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head, self.n_embd = n_head, n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd); self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd); self.mlp = MLP(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
# ===== END unmodified pytorch code =====

def run():
    np.random.seed(0)
    V, C, H, L, BS, B, Tn = 16, 32, 4, 2, 8, 16, 8
    model = GPT(V, C, H, L, BS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Xnp = np.random.randint(0, V, size=(B, Tn)).astype(np.int64)
    Ynp = np.random.randint(0, V, size=(B, Tn)).astype(np.int64).reshape(-1)
    idx = torch.tensor(Xnp)
    tgt = torch.tensor(Ynp)
    logits0 = model(idx)
    losses = []
    for step in range(120):
        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, V), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 30 == 0 or step == 119:
            losses.append(round(float(loss.item()), 3))
    out1 = model(idx).numpy()
    pred = out1.reshape(-1, V).argmax(1)
    # ---- state_dict save / load round-trip into a fresh model ----
    sd = model.state_dict()
    model2 = GPT(V, C, H, L, BS)
    model2.load_state_dict(sd)
    out2 = model2(idx).numpy()
    reload_err = float(np.abs(out1 - out2).max())
    r = {"backend": backend,
         "logits_shape": list(logits0.shape),
         "n_params": sum(int(p.data.size) for p in model.parameters()),
         "state_dict_keys": len(sd),
         "loss_curve": losses,
         "acc": round(float((pred == Ynp).mean()), 3),
         "reload_max_err": reload_err,
         "ran_unmodified": True}
    r["ok"] = bool(losses[-1] < losses[0] * 0.5 and r["acc"] > 0.85 and reload_err < 1e-4)
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)

run()
