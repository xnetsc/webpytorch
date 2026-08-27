"""Generic transformer-LM decode engine (config-driven, reusable).

Covers the CausalLM *series* — any Qwen2/Qwen3/Llama-style decoder: RMSNorm +
GQA attention (HF rope) + SwiGLU/MLP or MoE, int4/int8 QuantizedLinear weights,
a growing KV cache, and a pluggable sampler. It is not tied to any one model:
weights + config are supplied by a loader, the head + input embeddings are
pluggable, and samplers are generic (greedy / nucleus / ras).

Used by both plain text LLMs and CosyVoice2's speech-token LM (embedding input +
custom head + ras sampling) without any model-specific code in the engine.
"""
import math
import numpy as np
from . import _core as wt

xp = wt.xp


# ----------------------------- samplers (generic) -----------------------------
def sample_greedy(logits, *a, **k):
    return int(np.asarray(logits).argmax())

def _softmax_np(x):
    e = np.exp(x - x.max()); return e / e.sum()

def sample_nucleus(logits, top_p=0.8, top_k=25, rng=None, **k):
    """Nucleus (top-p) sampling, optionally capped at top-k.

    Only the head of the distribution is ever sampled from, so only the head is sorted. This
    used to argsort the whole vocabulary to look at the forty entries top-k asked for, and on
    a 150k-token vocabulary that one call was 32.8ms of the function's 43.5 -- more than four
    decode steps of the model it was sampling for, per token, and it did not shrink for
    smaller models because it scales with the vocabulary rather than the weights.
    `argpartition` finds the same head in 0.8ms and the sort then runs on the head alone.

    The head is widened rather than truncated if top-p is not reached inside it, so the
    result is what a full sort would have given, not an approximation of it."""
    lg = np.asarray(logits, np.float32)
    V = lg.size
    kk = V if (top_k is None or top_k <= 0) else min(int(top_k), V)
    # The exponentials, but NOT the division: top-p is a fraction of the total mass, so the
    # normaliser has to include the tail, yet only the head is ever divided by it. Dividing
    # the whole vocabulary is another full pass for values that are thrown away, and the
    # ordering is the same either way, so the partition runs on the exponentials directly.
    ex = np.exp(lg - lg.max())
    Z = float(ex.sum())
    # Start from the head top-k asks for, or 256 when it asks for no bound at all -- any p a
    # caller would pass is reached well inside that, and the loop widens if it is not. One
    # past top-k, so a head that top-k fills exactly still satisfies the exit test.
    n = min(V, (kk + 1) if kk < V else 256)
    while True:
        head = np.argpartition(ex, V - n)[V - n:] if n < V else np.arange(V)
        order = head[np.argsort(-ex[head])]
        # float64 for the running sum only: it is a sum of up to V terms and the cut is a
        # comparison against it, while the probabilities themselves do not need the width.
        cum = np.cumsum(ex[order], dtype=np.float64) / Z
        # The token that crosses top_p is kept, matching the usual definition (and what a
        # cumulative loop that tests before adding produces).
        m = min(kk, int(np.searchsorted(cum, top_p)) + 1, n)
        if m < n or n >= V:
            break
        # The head was not enough, so the tail genuinely carries mass -- a flat distribution,
        # which is what a high temperature produces. Go straight to the whole vocabulary
        # rather than widening by steps: each step repeats the partition and the sort, and
        # geometric widening on a flat 150k vocabulary measured 59.9ms against the 38.1 of
        # the full sort it was trying to avoid. Two rounds at worst, and the worst round is
        # still cheaper than what this replaced.
        n = V
    keep = order[:m]
    pk = ex[keep].astype(np.float64)
    pk /= pk.sum()
    r = (rng or np.random).random()
    return int(keep[np.searchsorted(np.cumsum(pk), r)])

def sample_ras(logits, decoded, top_p=0.8, top_k=25, win_size=10, tau_r=0.1, rng=None, **k):
    """CosyVoice repetition-aware sampling: nucleus, but if the pick repeats too
    often in the recent window, blacklist it and fall back to a plain random draw."""
    tid = sample_nucleus(logits, top_p, top_k, rng)
    recent = decoded[-win_size:]
    if recent and sum(1 for t in recent if t == tid) >= win_size * tau_r:
        lg = np.asarray(logits, np.float64).copy(); lg[tid] = -np.inf
        p = _softmax_np(lg)
        tid = int((rng or np.random).choice(len(p), p=p))
    return tid

SAMPLERS = {"greedy": sample_greedy, "nucleus": sample_nucleus, "ras": sample_ras}


# ----------------------------- generic LM -----------------------------
class TransformerLM:
    """Config-driven decoder. `layers` is a list of dicts of QuantizedLinear
    (q,k,v,o,gate,up,down) + Tensor norms (in_ln,post_ln); optional MoE fields
    (see moe_mlp). Weights/config come from a loader; this class only runs it."""

    def __init__(self, cfg):
        self.H = cfg["H"]; self.L = cfg["L"]; self.NH = cfg["NH"]; self.NKV = cfg["NKV"]
        self.HD = cfg["HD"]; self.eps = cfg["eps"]; self.theta = cfg["theta"]
        self.layers = []; self.final_norm = None
        self.head = None            # callable: hidden(1,H) -> logits(V,)
        self.embed_next = None      # callable: token_id -> (H,) embedding for the next step

    # ---- math ----
    def _rms(self, x, w):
        return (x / ((x * x).mean(axis=-1, keepdims=True) + self.eps).sqrt()) * w

    def _rope(self, pos, T=1):
        inv = 1.0 / (self.theta ** (np.arange(0, self.HD, 2, dtype=np.float64) / self.HD))
        ang = np.arange(pos, pos + T, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([ang, ang], -1)
        return wt.Tensor(np.cos(emb).astype(np.float32)), wt.Tensor(np.sin(emb).astype(np.float32))

    def _rot(self, x):
        hd = self.HD
        return wt.cat([wt._slice_last(x, hd // 2, hd) * (-1.0), wt._slice_last(x, 0, hd // 2)], axis=-1)

    def _mlp(self, lay, x):
        if lay.get("moe"):
            return moe_mlp(self, lay, x)
        return lay["down"](_swiglu(lay["gate"](x), lay["up"](x)))

    def _block(self, i, h, cos, sin, K, V, mask):
        H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        lay = self.layers[i]; T = h.shape[0]; sc = 1.0 / math.sqrt(HD)
        x = self._rms(h, lay["in_ln"])
        q = lay["q"](x).reshape(T, NH, HD).permute(1, 0, 2)
        k = lay["k"](x).reshape(T, NKV, HD).permute(1, 0, 2)
        v = lay["v"](x).reshape(T, NKV, HD).permute(1, 0, 2)
        q = q * cos + self._rot(q) * sin
        k = k * cos + self._rot(k) * sin
        # grow cache
        if K[i] is None:
            K[i] = k; V[i] = v
        else:
            K[i] = wt.cat([K[i], k], axis=1); V[i] = wt.cat([V[i], v], axis=1)
        o = wt.gqa_attention(q, K[i], V[i], mask, scale=sc).permute(1, 0, 2).reshape(T, H)
        h = h + lay["o"](o)
        x = self._rms(h, lay["post_ln"])
        h = h + self._mlp(lay, x)
        return h

    def prefill(self, embs):
        """embs: (T,H) Tensor prompt embeddings -> (last-hidden Tensor, KV state, pos)."""
        T = embs.shape[0]
        cos, sin = self._rope(0, T)
        mask = wt.Tensor(np.triu(np.full((T, T), -1e9, np.float32), 1))   # (T,T) broadcasts in gqa
        self.K = [None] * self.L; self.V = [None] * self.L
        h = embs
        for i in range(self.L):
            h = self._block(i, h, cos, sin, self.K, self.V, mask)
        h = self._rms(h, self.final_norm)
        last = wt.Tensor(wt._contig(h.data[-1:]))
        return last, T

    def step(self, emb, pos):
        """emb: (1,H) Tensor next-token embedding at position `pos` -> (1,hidden)."""
        cos, sin = self._rope(pos, 1)
        h = emb
        for i in range(self.L):
            h = self._block(i, h, cos, sin, self.K, self.V, None)
        return self._rms(h, self.final_norm)

    # ---- capture-accelerated decode (WebGPU): fixed KV cache + replay (~20x) ----
    def _rope_np(self, pos, T=1):
        inv = 1.0 / (self.theta ** (np.arange(0, self.HD, 2, dtype=np.float64) / self.HD))
        ang = np.arange(pos, pos + T, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([ang, ang], -1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def init_capture(self, lmax=512):
        self.lmax = lmax; L, NKV, HD, H = self.L, self.NKV, self.HD, self.H
        self._cap = wt._adam_backend_ready()
        if not self._cap:
            return False
        self.Kc = [wt.Tensor(wt._zeros((NKV, lmax, HD))) for _ in range(L)]
        self.Vc = [wt.Tensor(wt._zeros((NKV, lmax, HD))) for _ in range(L)]
        self.h_in = wt.Tensor(np.zeros((1, H), np.float32))
        self.cos_b = wt.Tensor(np.zeros((1, HD), np.float32))
        self.sin_b = wt.Tensor(np.zeros((1, HD), np.float32))
        self.mask_b = wt.Tensor(np.zeros((1, 1, lmax), np.float32))
        self.ctl = xp.asarray(np.array([0, 1, NKV, HD, lmax], np.int32))
        return True

    def _prefill_fixed(self, embs):
        T = embs.shape[0]; NH, NKV, HD, LMAX, H = self.NH, self.NKV, self.HD, self.lmax, self.H
        c, s = self._rope_np(0, T); cos_t, sin_t = wt.Tensor(c), wt.Tensor(s)
        m = np.triu(np.full((T, LMAX), -1e9, np.float32), 1); m[:, T:] = -1e9
        mask = wt.Tensor(m); sc = 1.0 / math.sqrt(HD); h = embs
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            q = lay["q"](x).reshape(T, NH, HD).permute(1, 0, 2)
            k = lay["k"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            v = lay["v"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            q = q * cos_t + self._rot(q) * sin_t; k = k * cos_t + self._rot(k) * sin_t
            self.Kc[i].data = wt.kv_write(self.Kc[i].data, wt._contig(k).data, 0, T, NKV, HD, LMAX)
            self.Vc[i].data = wt.kv_write(self.Vc[i].data, wt._contig(v).data, 0, T, NKV, HD, LMAX)
            o = wt.gqa_attention(q, self.Kc[i], self.Vc[i], mask, scale=sc)
            h = h + lay["o"](o.permute(1, 0, 2).reshape(T, H))
            x = self._rms(h, lay["post_ln"]); h = h + self._mlp(lay, x)
        last = wt.Tensor(wt._contig(self._rms(h, self.final_norm).data[-1:]))
        return last

    def _set_inputs(self, emb_vec, pos):
        NKV, HD, LMAX = self.NKV, self.HD, self.lmax
        self.h_in.data.buffer.set_data(np.asarray(emb_vec, np.float32))
        c, s = self._rope_np(pos)
        self.cos_b.data.buffer.set_data(c.reshape(-1)); self.sin_b.data.buffer.set_data(s.reshape(-1))
        m = np.zeros((1, 1, LMAX), np.float32); m[0, 0, pos + 1:] = -1e9
        self.mask_b.data.buffer.set_data(m)
        self.ctl.buffer.set_data(np.array([pos, 1, NKV, HD, LMAX], np.int32))

    def _decode_fixed(self):
        NH, NKV, HD, LMAX, H = self.NH, self.NKV, self.HD, self.lmax, self.H
        sc = 1.0 / math.sqrt(HD); h = self.h_in
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            q = lay["q"](x).reshape(1, NH, HD).permute(1, 0, 2)
            k = lay["k"](x).reshape(1, NKV, HD).permute(1, 0, 2)
            v = lay["v"](x).reshape(1, NKV, HD).permute(1, 0, 2)
            q = q * self.cos_b + self._rot(q) * self.sin_b
            k = k * self.cos_b + self._rot(k) * self.sin_b
            self.Kc[i].data = wt.kv_write(self.Kc[i].data, wt._contig(k).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
            self.Vc[i].data = wt.kv_write(self.Vc[i].data, wt._contig(v).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
            o = wt.gqa_attention(q, self.Kc[i], self.Vc[i], self.mask_b, scale=sc)
            h = h + lay["o"](o.permute(1, 0, 2).reshape(1, H))
            x = self._rms(h, lay["post_ln"]); h = h + self._mlp(lay, x)
        return self.head(self._rms(h, self.final_norm))

    def generate_captured(self, prompt_embs, max_new, stop_ids, sampler="greedy",
                          sampler_kwargs=None, seed=0, min_new=0):
        """WebGPU capture-replay decode: capture one step, replay per token (~20x).
        Sampling runs on host over the replayed logits (works with any sampler, incl ras)."""
        skw = dict(sampler_kwargs or {}); rng = np.random.RandomState(seed)
        fn = SAMPLERS[sampler]; stop_ids = set(stop_ids); stop_arr = np.array(sorted(stop_ids), np.int64)
        for c in self.Kc: c.data[:] = 0.0
        for v in self.Vc: v.data[:] = 0.0
        last = self._prefill_fixed(prompt_embs); P = prompt_embs.shape[0]
        plat = wt._adam_kernel["platform"]
        logits0 = self.head(last)                    # first logits (prefill hidden)
        out = []; pos = P
        # capture a decode step
        first = logits0.numpy().reshape(-1)
        def pick(lg, step):
            if step < min_new: lg = lg.copy(); lg[stop_arr] = -np.inf
            return fn(lg, out, rng=rng, **skw)
        tid = pick(first, 0)
        captured = False
        for step in range(max_new):
            if tid in stop_ids: break
            out.append(tid)
            self._set_inputs(self.embed_next(tid).reshape(-1), pos); pos += 1
            if not captured:
                plat.beginCapture("lm_decode"); logits_t = self._decode_fixed(); logits_t.numpy()
                plat.endCapture(); captured = True
            else:
                plat.replay("lm_decode")
            lg = logits_t.numpy().reshape(-1)
            tid = pick(lg, step + 1)
        return out

    def generate_stream(self, prompt_ids, max_new, stop_ids, sampler="greedy",
                        sampler_kwargs=None, seed=0, min_new=0, embed=None):
        """STREAMING output: yield each decoded token id as it is produced (UI can
        render live; memory stays bounded). `prompt_ids` = token ids; `embed` (id->vec)
        defaults to self.embed_next. Uses capture-replay if init_capture() ran."""
        embed = embed or self.embed_next
        skw = dict(sampler_kwargs or {}); rng = np.random.RandomState(seed)
        fn = SAMPLERS[sampler]; stop_ids = set(stop_ids); stop_arr = np.array(sorted(stop_ids), np.int64)
        prompt_embs = wt.Tensor(np.stack([np.asarray(embed(int(i)), np.float32) for i in prompt_ids]))
        cap = getattr(self, "_cap", False)

        def pick(lg, step):
            if step < min_new: lg = lg.copy(); lg[stop_arr] = -np.inf
            return fn(lg, out, rng=rng, **skw)
        out = []
        if not cap:                                    # growing-cache path
            last, pos = self.prefill(prompt_embs)
            for step in range(max_new):
                tid = pick(self.head(last).numpy().reshape(-1), step)
                if tid in stop_ids: break
                out.append(tid); yield tid
                last = self.step(wt.Tensor(np.asarray(embed(tid), np.float32).reshape(1, -1)), pos); pos += 1
            return
        # capture-replay path: head is inside the captured decode graph
        for c in self.Kc: c.data[:] = 0.0
        for v in self.Vc: v.data[:] = 0.0
        last = self._prefill_fixed(prompt_embs); P = prompt_embs.shape[0]
        plat = wt._adam_kernel["platform"]
        tid = pick(self.head(last).numpy().reshape(-1), 0); pos = P; captured = False
        for step in range(max_new):
            if tid in stop_ids: break
            out.append(tid); yield tid
            self._set_inputs(np.asarray(embed(tid), np.float32).reshape(-1), pos); pos += 1
            if not captured:
                plat.beginCapture("lm_decode"); logits_t = self._decode_fixed(); logits_t.numpy()
                plat.endCapture(); captured = True
            else:
                plat.replay("lm_decode")
            tid = pick(logits_t.numpy().reshape(-1), step + 1)

    def generate(self, prompt_embs, max_new, stop_ids, sampler="greedy",
                 sampler_kwargs=None, seed=0, min_new=0):
        """Autoregressive decode from prompt embeddings. Returns list of token ids.
        Uses self.head (hidden->logits) and self.embed_next (id->next embedding).
        For the first `min_new` steps, stop tokens are masked out (ignore-eos), so
        generation cannot terminate prematurely (mirrors HF min_new_tokens)."""
        skw = dict(sampler_kwargs or {}); rng = np.random.RandomState(seed)
        fn = SAMPLERS[sampler]; stop_ids = set(stop_ids)
        stop_arr = np.array(sorted(stop_ids), np.int64)
        last, pos = self.prefill(prompt_embs)
        out = []
        for step in range(max_new):
            logits = self.head(last).numpy().reshape(-1)
            if step < min_new:
                logits = logits.copy(); logits[stop_arr] = -np.inf
            tid = fn(logits, out, rng=rng, **skw)
            if tid in stop_ids:
                break
            out.append(tid)
            emb = wt.Tensor(self.embed_next(tid).reshape(1, -1).astype(np.float32))
            last = self.step(emb, pos)
            pos += 1
        return out


# ----------------------------- generic MoE (series-level) -----------------------------
def _swiglu(gate, up=None):
    """The SwiGLU activation, fused when the device can and the graph form otherwise.

    One dispatch instead of six, and with a fused gate/up weight the two halves are read in
    place rather than sliced into their own buffers first."""
    r = wt.swiglu(gate, up)
    if r is not None:
        return r
    if up is None:
        half = gate.data.shape[-1] // 2
        return wt.silu(gate[:, :half]) * gate[:, half:]
    return wt.silu(gate) * up


def moe_mlp(lm, lay, x):
    """Generic sparse-MoE MLP: router top-k over experts (+ optional shared expert,
    Qwen-style). `lay['moe']` = dict(gate, experts=[{gate,up,down}], top_k,
    norm_topk_prob, shared=(gate,up,down)|None, shared_gate=QLinear|None).
    Runs on host for routing (small) + QuantizedLinear experts on GPU."""
    m = lay["moe"]; T = x.shape[0]
    k = int(m["top_k"])
    st = m.get("stacked")
    norm = bool(m.get("norm_topk_prob", m.get("norm_topk", True)))
    rlog = m["gate"](x)                                   # (T, n_experts), on device
    if T == 1 and st is not None and getattr(wt, "moe_route", None) is not None:
        # Everything on the device: the router's scores go straight into the index and weight
        # buffers the expert matmuls read, so a decode step makes no round-trip at all. On a
        # 48-layer model, reading the router back once per layer WAS most of the step.
        ne = int(m.get("n_experts") or st["gate"].n_experts)
        buf = m.get("_gpu_route")
        if buf is None:
            buf = {"eidx": wt._empty_i32((k,)), "ew": wt.Tensor(np.zeros((k,), np.float32))}
            m["_gpu_route"] = buf
        wt.moe_route(rlog.data, buf["eidx"], buf["ew"].data, ne, k, norm)
        eidx = buf["eidx"]
        gu = st["gate_up"].forward(x, eidx)               # (k, 2*inter): gate then up
        y = st["down"].forward(_swiglu(gu), eidx)
        out = (y * buf["ew"].reshape(k, 1)).sum(axis=0).reshape(1, -1)
        if m.get("shared"):
            sh = m["shared"]
            ys = sh["down"](_swiglu(sh["gate"](x), sh["up"](x)))
            if m.get("shared_gate") is not None:
                ys = ys * m["shared_gate"](x).sigmoid()
            out = out + ys
        return out
    # Paths that still decide on the host need the scores there.
    router_logits = rlog.numpy()                          # (T, n_experts)
    ne = router_logits.shape[1]
    # top-k gates (softmax over full set, then renorm over top-k like Qwen)
    probs = _softmax_rows(router_logits)
    topk_idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    topk_w = np.take_along_axis(probs, topk_idx, axis=1)
    if norm:
        topk_w = topk_w / topk_w.sum(1, keepdims=True)
    if T == 1 and st is not None:
        # Decode with the experts stacked: the chosen indices go into a small buffer and the
        # dispatches are the same commands every token, whichever experts the router picked.
        # That is what keeps the step capturable -- the alternative bakes the first token's
        # routing into the command list and repeats it forever.
        eidx = m.get("_eidx")
        if eidx is None or int(eidx.size) < k:
            eidx = wt._empty_i32((max(k, 1),))
            m["_eidx"] = eidx
        eidx.buffer.set_data(np.ascontiguousarray(topk_idx[0, :k].astype(np.int32)))
        wsel = m.get("_w")
        if wsel is None or int(wsel.data.size) < k:
            wsel = wt.Tensor(np.zeros((k,), np.float32))
            m["_w"] = wsel
        wsel.data.buffer.set_data(np.ascontiguousarray(topk_w[0, :k].astype(np.float32)))
        # Three dispatches for the whole layer, not three per slot: each covers all k routed
        # experts at once (z indexes the slot), which is the difference between a command
        # with enough work to fill the GPU and k commands that each leave it idle.
        g = st["gate"].forward(x, eidx)            # (k, inter)
        u = st["up"].forward(x, eidx)
        y = st["down"].forward(_swiglu(g, u), eidx)   # (k, H)
        out = (y * wsel.reshape(k, 1)).sum(axis=0).reshape(1, -1)
    elif T == 1:
        # Decode: one token, so every selected expert contributes to the same single row.
        # Walking all `ne` experts to find which were picked, and scattering each result back
        # with a fancy-index assignment, costs a host round-trip per expert for a routing
        # decision already made -- and this is the step that runs once per token. Sum the k
        # selected experts directly instead; nothing leaves the device.
        acc = None
        for slot in range(k):
            exp = m["experts"][int(topk_idx[0, slot])]
            ye = exp["down"](_swiglu(exp["gate"](x), exp["up"](x)))
            ye = ye * float(topk_w[0, slot])
            acc = ye if acc is None else acc + ye
        out = acc
    elif st is not None:
        # Prefill, one token at a time through the same batched form. Gathering tokens per
        # expert would cut the work, but a prompt is a handful of steps against a decode's
        # thousands, and this keeps one code path.
        eidx = wt._empty_i32((k,))
        rows = []
        for ti in range(T):
            xt = wt.Tensor(wt._contig(x.data[ti:ti + 1]))
            eidx.buffer.set_data(np.ascontiguousarray(topk_idx[ti, :k].astype(np.int32)))
            gu = st["gate_up"].forward(xt, eidx)
            y = st["down"].forward(_swiglu(gu), eidx)
            wv = wt.Tensor(np.ascontiguousarray(topk_w[ti, :k].astype(np.float32)))
            rows.append((y * wv.reshape(k, 1)).sum(axis=0).reshape(1, -1))
        out = wt.cat(rows, axis=0)
    else:
        out = wt.Tensor(np.zeros((T, lm.H), np.float32))
        # gather tokens per expert (host indices), run each selected expert once
        for e in range(ne):
            rows, slot = np.where(topk_idx == e)
            if len(rows) == 0:
                continue
            xe = wt.Tensor(wt._contig(x.data[rows]))      # (m, H)
            exp = m["experts"][e]
            ye = exp["down"](_swiglu(exp["gate"](xe), exp["up"](xe)))
            w = topk_w[rows, slot].astype(np.float32)
            ye = ye * wt.Tensor(w.reshape(-1, 1))
            acc = out.data
            acc[rows] = acc[rows] + ye.data
            out = wt.Tensor(acc)
    if m.get("shared"):
        s = m["shared"]
        ys = s["down"](_swiglu(s["gate"](x), s["up"](x)))
        if m.get("shared_gate") is not None:
            g = wt.Tensor(1.0 / (1.0 + np.exp(-m["shared_gate"](x).numpy())))
            ys = ys * g
        out = out + ys
    return out

def _softmax_rows(x):
    x = np.asarray(x, np.float64); e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ----------------------------- generic builder (dense + MoE series) -----------------------------
def build_lm(cfg, get, linear, tensor):
    """Construct a TransformerLM from a weight source, dense OR MoE, config-driven.

      cfg    : dict with H,L,NH,NKV,HD,eps,theta (+ optional num_experts, num_experts_per_tok,
               norm_topk_prob, shared_expert (bool), decoder_sparse_step, mlp_only_layers, head/embed).
      get(name)  -> np weight for a param name (streams from safetensors/gguf/npz).
      linear(Wname, bias=True) -> a callable (Tensor -> Tensor) for that weight (int4 QuantizedLinear
               in the browser, fp32 matmul offline). `bias=True` also loads Wname[:-7]+'.bias' if present.
      tensor(np) -> wt.Tensor.

    Returns a ready TransformerLM. This is the single generic loader behind the whole
    CausalLM + MoE series — no per-model code."""
    lm = TransformerLM(cfg)
    L = cfg["L"]; ne = cfg.get("num_experts", 0)
    sparse_step = cfg.get("decoder_sparse_step", 1); only = set(cfg.get("mlp_only_layers", []))
    P = cfg.get("layer_prefix", "model.layers.")
    for i in range(L):
        p = f"{P}{i}."
        lay = {"in_ln": tensor(get(p + "input_layernorm.weight")),
               "post_ln": tensor(get(p + "post_attention_layernorm.weight")),
               "q": linear(p + "self_attn.q_proj.weight"), "k": linear(p + "self_attn.k_proj.weight"),
               "v": linear(p + "self_attn.v_proj.weight"), "o": linear(p + "self_attn.o_proj.weight", bias=False)}
        is_moe = ne > 0 and (i not in only) and ((i + 1) % sparse_step == 0)
        if is_moe:
            moe = {"gate": linear(p + "mlp.gate.weight", bias=False), "top_k": cfg["num_experts_per_tok"],
                   "norm_topk_prob": cfg.get("norm_topk_prob", True),
                   "experts": [{"gate": linear(p + f"mlp.experts.{e}.gate_proj.weight", bias=False),
                                "up": linear(p + f"mlp.experts.{e}.up_proj.weight", bias=False),
                                "down": linear(p + f"mlp.experts.{e}.down_proj.weight", bias=False)} for e in range(ne)],
                   "shared": None}
            if cfg.get("shared_expert"):
                moe["shared"] = {"gate": linear(p + "mlp.shared_expert.gate_proj.weight", bias=False),
                                 "up": linear(p + "mlp.shared_expert.up_proj.weight", bias=False),
                                 "down": linear(p + "mlp.shared_expert.down_proj.weight", bias=False)}
                moe["shared_gate"] = linear(p + "mlp.shared_expert_gate.weight", bias=False)
            lay["moe"] = moe
        else:
            lay["gate"] = linear(p + "mlp.gate_proj.weight", bias=False)
            lay["up"] = linear(p + "mlp.up_proj.weight", bias=False)
            lay["down"] = linear(p + "mlp.down_proj.weight", bias=False)
        lm.layers.append(lay)
    lm.final_norm = tensor(get(cfg.get("final_norm", "model.norm.weight")))
    return lm
