"""CosyVoice2-0.5B flow + vocoder in the browser (host numpy, Pyodide).

Ports the numpy-validated CosyVoice2 pipeline: speech tokens -> UpsampleConformerEncoder
-> encoder_proj -> CausalConditionalCFM (10-step Euler + CFG, estimator UNet) -> mel
-> HiFTGenerator (f0 + NSF source + STFT/ISTFT) -> 24 kHz waveform.

Validated offline vs native CosyVoice2 (torch 2.3.1): flow mel rel 1.8e-6, vocoder
decode 4.7e-6. The autoregressive Qwen2-0.5B LLM (text -> speech tokens) is NOT here yet;
this module runs on baked/provided speech tokens (fixed built-in speaker path). fp32.
"""
import io, math, json
import numpy as np
try:
    from . import _core as wt   # GPU (WgPy WebGPU/WebGL) tensor engine; numpy fallback offline
except Exception:
    wt = None

# ---------- shared numeric helpers ----------
def _silu(x): return x / (1.0 + np.exp(-x))
def _swish(x): return x / (1.0 + np.exp(-x))
def _mish(x):
    sp = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    return x * np.tanh(sp)
def _elu(x): return np.where(x > 0, x, np.exp(np.minimum(x, 0.0)) - 1)
def _lrelu(x, s=0.01): return np.where(x >= 0, x, x * s)
def _erf(x):
    # Abramowitz & Stegun 7.1.26, |err| < 1.5e-7
    s = np.sign(x); ax = np.abs(x); t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return s * y
def _gelu(x): return x * 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
def _layernorm(x, w, b, eps=1e-5, axis=-1):
    mu = x.mean(axis=axis, keepdims=True); v = x.var(axis=axis, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + b
def _hann(n):
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)

def _conv1d(x, w, b, s=1, pad=0, dil=1):
    # x: (C,L) or (B,C,L); w: (O,Cin,K). Batched, float32.
    single = (x.ndim == 2)
    if single: x = x[None]
    B, C, L = x.shape; O, _, K = w.shape
    if pad: x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    Lp = x.shape[2]; eff = (K - 1) * dil + 1; Lo = (Lp - eff) // s + 1
    if C * K * B * Lo <= (96 << 20):    # one big im2col GEMM (fast); accumulate only if the buffer is huge
        idx = s * np.arange(Lo)[:, None] + dil * np.arange(K)[None, :]
        colr = x[:, :, idx].transpose(1, 3, 0, 2).reshape(C * K, B * Lo)
        out = (w.reshape(O, C * K) @ colr).reshape(O, B, Lo).transpose(1, 0, 2)
    else:                               # large: accumulate per-tap (memory-light for the vocoder)
        wr = w.reshape(O, C, K); out = np.zeros((B, O, Lo), np.float32)
        for k in range(K):
            xs = x[:, :, k * dil:k * dil + s * Lo:s]                   # (B,C,Lo)
            out += (wr[:, :, k] @ xs.transpose(1, 0, 2).reshape(C, B * Lo)).reshape(O, B, Lo).transpose(1, 0, 2)
    if b is not None: out += b[None, :, None]
    return out[0] if single else out
def _causal_conv1d(x, w, b, k=3):  # left-pad k-1 on the time axis (2D or 3D)
    pw = ((0, 0),) * (x.ndim - 1) + ((k - 1, 0),)
    return _conv1d(np.pad(x, pw), w, b)
def _convT1d(x, w, b, stride, pad):
    Ci, L = x.shape; _, O, K = w.shape; Lout = (L - 1) * stride + K
    full = np.zeros((O, Lout), np.float32)
    for k in range(K): full[:, k:k + stride * L:stride] += w[:, :, k].T @ x
    out = full[:, pad:full.shape[1] - pad] if pad else full
    if b is not None: out += b[:, None]
    return out


class CosyVoice2TTS:
    def __init__(self, flowZ, hiftZ, baked):
        _d = lambda z: {k: z[k] for k in (z.files if hasattr(z, 'files') else z)}   # npz or dict
        self.F = _d(flowZ); self.H = _d(hiftZ); self.baked = _d(baked)
        self.sr = 24000

    @classmethod
    async def from_npz(cls, flow, hift, baked, io=None):
        # each arg: url (str) | bytes | reader callback — injected IO via webio
        from . import webio
        return cls(await webio.load_npz(flow, io), await webio.load_npz(hift, io),
                   await webio.load_npz(baked, io))

    # ================= FLOW =================
    def _f(self, k): return self.F[k]
    def _lin(self, x, p, bias=True):
        w = self.F[p + '.weight']; y = x @ w.T
        if bias and p + '.bias' in self.F: y = y + self.F[p + '.bias']
        return y

    # ---- UpsampleConformerEncoder ----
    def _relpos(self, T, D=512):
        pos = np.arange(T)[:, None].astype(np.float64)
        div = np.exp(np.arange(0, D, 2) * -(math.log(10000.0) / D))
        # float32: np.zeros defaults to float64, which is two bytes a value too many for a
        # table that is only ever multiplied into float32 activations.
        pep = np.zeros((T, D), np.float32)
        pep[:, 0::2] = np.sin(pos * div); pep[:, 1::2] = np.cos(pos * div)
        pen = np.zeros((T, D), np.float32)
        pen[:, 0::2] = np.sin(-pos * div); pen[:, 1::2] = np.cos(-pos * div)
        return np.concatenate([pep[::-1], pen[1:]], 0).astype(np.float32)
    def _rel_shift(self, x):
        Hh, T, P = x.shape
        x = np.concatenate([np.zeros((Hh, T, 1), x.dtype), x], -1)
        x = x.reshape(Hh, P + 1, T); x = x[:, 1:].reshape(Hh, T, P)
        return x[:, :, :P // 2 + 1]
    def _enc_attn(self, x, pe, p, H=8, DK=64, D=512):
        T = x.shape[0]
        q = self._lin(x, p + '.linear_q').reshape(T, H, DK)
        k = self._lin(x, p + '.linear_k').reshape(T, H, DK).transpose(1, 0, 2)
        v = self._lin(x, p + '.linear_v').reshape(T, H, DK).transpose(1, 0, 2)
        pp = self._lin(pe, p + '.linear_pos', bias=False).reshape(-1, H, DK).transpose(1, 0, 2)
        bu = self.F[p + '.pos_bias_u']; bv = self.F[p + '.pos_bias_v']
        qu = (q + bu).transpose(1, 0, 2); qv = (q + bv).transpose(1, 0, 2)
        ac = qu @ k.transpose(0, 2, 1); bd = self._rel_shift(qv @ pp.transpose(0, 2, 1))
        sc = (ac + bd) / math.sqrt(DK)
        sc = sc - sc.max(-1, keepdims=True); a = np.exp(sc); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(T, D)
        return self._lin(o, p + '.linear_out')
    def _enc_block(self, x, pe, p):
        x = x + self._enc_attn(_layernorm(x, self.F[p + '.norm_mha.weight'], self.F[p + '.norm_mha.bias'], 1e-12), pe, p + '.self_attn')
        ff = self._lin(_swish(self._lin(_layernorm(x, self.F[p + '.norm_ff.weight'], self.F[p + '.norm_ff.bias'], 1e-12), p + '.feed_forward.w_1')), p + '.feed_forward.w_2')
        return x + ff
    def _enc_embed(self, x, pfx, D=512):
        x = _layernorm(self._lin(x, pfx + '.out.0'), self.F[pfx + '.out.1.weight'], self.F[pfx + '.out.1.bias'], 1e-5)
        return x * math.sqrt(D), self._relpos(x.shape[0])
    def _encoder(self, emb):
        x, pe = self._enc_embed(emb, 'encoder.embed')
        xt = x.T; xt = np.pad(xt, ((0, 0), (0, 3)))
        xt = _lrelu(_conv1d(xt, self.F['encoder.pre_lookahead_layer.conv1.weight'], self.F['encoder.pre_lookahead_layer.conv1.bias']), 0.01)
        xt = np.pad(xt, ((0, 0), (2, 0)))
        xt = _conv1d(xt, self.F['encoder.pre_lookahead_layer.conv2.weight'], self.F['encoder.pre_lookahead_layer.conv2.bias'])
        x = x + xt.T
        for i in range(6): x = self._enc_block(x, pe, f'encoder.encoders.{i}')
        xt = x.T; xt = np.repeat(xt, 2, axis=1); xt = np.pad(xt, ((0, 0), (4, 0)))
        xt = _conv1d(xt, self.F['encoder.up_layer.conv.weight'], self.F['encoder.up_layer.conv.bias'])
        x = xt.T
        x, pe = self._enc_embed(x, 'encoder.up_embed')
        for i in range(4): x = self._enc_block(x, pe, f'encoder.up_encoders.{i}')
        return _layernorm(x, self.F['encoder.after_norm.weight'], self.F['encoder.after_norm.bias'], 1e-12)

    # ============ GPU conformer encoder (webtorch Tensors, single forward) ============
    def _et(self, k):
        c = self.__dict__.setdefault('_etraw', {})
        if k not in c: c[k] = self._wtT(self.F[k])
        return c[k]
    def _etl(self, x, p, bias=True):
        c = self.__dict__.setdefault('_etlin', {})
        if p not in c: c[p] = self._wtT(self.F[p + '.weight'].T)
        y = x.matmul(c[p])
        if bias and p + '.bias' in self.F: y = y + self._et(p + '.bias')
        return y
    def _etconv(self, x, p):                 # x (C,T) explicit-padded -> (O, T-K+1)
        c = self.__dict__.setdefault('_etcv', {})
        if p not in c:
            w = self.F[p + '.weight']; c[p] = [self._wtT(w[:, :, k].T) for k in range(w.shape[2])]
        taps = c[p]; C, T = x.shape; K = len(taps); Lo = T - K + 1
        out = None
        for k in range(K):
            y = wt._slice_last(x, k, k + Lo).permute(1, 0).matmul(taps[k])   # (Lo,O)
            out = y if out is None else out + y
        out = out.permute(1, 0)              # (O,Lo)
        if p + '.bias' in self.F: out = out + self._et(p + '.bias').reshape(-1, 1)
        return out
    @staticmethod
    def _we_lrelu(x, s=0.01):
        return x.relu() + (x * (-1.0)).relu() * (-s)
    def _relpos_gpu(self, T):
        c = self.__dict__.setdefault('_relc', {})
        if T not in c: c[T] = wt.Tensor(self._relpos(T))
        return c[T]
    def _we_rel_shift(self, xt):             # xt Tensor (H,T,P) P=2T-1 -> (H,T,T)
        d = xt.data; xp = wt.xp; H, T, P = d.shape
        z = xp.zeros((H, T, P + 1), np.float32); z[:, :, 1:] = d
        z = z.reshape(H, P + 1, T)
        z = wt._contig(z[:, 1:, :]).reshape(H, T, P)
        return wt.Tensor(wt._contig(z[:, :, :T]))
    def _we_attn(self, x, pe, p, H=8, DK=64, D=512):
        T = x.shape[0]
        q = self._etl(x, p + '.linear_q').reshape(T, H, DK)
        k = self._etl(x, p + '.linear_k').reshape(T, H, DK).permute(1, 0, 2)
        v = self._etl(x, p + '.linear_v').reshape(T, H, DK).permute(1, 0, 2)
        pp = self._etl(pe, p + '.linear_pos', bias=False)
        P = pp.shape[0]; pp = pp.reshape(P, H, DK).permute(1, 0, 2)
        bu = self._et(p + '.pos_bias_u'); bv = self._et(p + '.pos_bias_v')
        qu = (q + bu).permute(1, 0, 2); qv = (q + bv).permute(1, 0, 2)
        ac = wt.bmm(qu, k.permute(0, 2, 1))
        bd = self._we_rel_shift(wt.bmm(qv, pp.permute(0, 2, 1)))
        a = wt.softmax((ac + bd) * (1.0 / math.sqrt(DK)))
        o = wt.bmm(a, v).permute(1, 0, 2).reshape(T, D)
        return self._etl(o, p + '.linear_out')
    def _we_block(self, x, pe, p):
        n = wt.layernorm(x, self._et(p + '.norm_mha.weight'), self._et(p + '.norm_mha.bias'), eps=1e-12)
        x = x + self._we_attn(n, pe, p + '.self_attn')
        n = wt.layernorm(x, self._et(p + '.norm_ff.weight'), self._et(p + '.norm_ff.bias'), eps=1e-12)
        ff = self._etl(wt.silu(self._etl(n, p + '.feed_forward.w_1')), p + '.feed_forward.w_2')
        return x + ff
    def _we_embed(self, x, pfx, D=512):
        x = wt.layernorm(self._etl(x, pfx + '.out.0'), self._et(pfx + '.out.1.weight'), self._et(pfx + '.out.1.bias'), eps=1e-5)
        return x * math.sqrt(D), self._relpos_gpu(x.shape[0])
    def _zpad(self, x, left, right):         # x (C,T) -> zero-pad last axis
        xp = wt.xp; C, T = x.shape
        z = xp.zeros((C, T + left + right), np.float32); z[:, left:left + T] = x.data
        return wt.Tensor(z)
    def _encoder_gpu(self, emb):
        x, pe = self._we_embed(wt.Tensor(emb.astype(np.float32)), 'encoder.embed')
        xt = self._zpad(wt.Tensor(wt._contig(x.data.T)), 0, 3)
        xt = self._we_lrelu(self._etconv(xt, 'encoder.pre_lookahead_layer.conv1'), 0.01)
        xt = self._zpad(xt, 2, 0)
        xt = self._etconv(xt, 'encoder.pre_lookahead_layer.conv2')
        x = x + wt.Tensor(wt._contig(xt.data.T))
        for i in range(6): x = self._we_block(x, pe, f'encoder.encoders.{i}')
        # up_layer: nearest x2 upsample (strided assign), pad-left 4, conv k5
        xp = wt.xp; xd = wt._contig(x.data.T); C, T = xd.shape
        up = xp.zeros((C, 2 * T), np.float32); up[:, 0::2] = xd; up[:, 1::2] = xd
        xt = self._zpad(wt.Tensor(up), 4, 0)
        xt = self._etconv(xt, 'encoder.up_layer.conv')
        x = wt.Tensor(wt._contig(xt.data.T))
        x, pe = self._we_embed(x, 'encoder.up_embed')
        for i in range(4): x = self._we_block(x, pe, f'encoder.up_encoders.{i}')
        x = wt.layernorm(x, self._et('encoder.after_norm.weight'), self._et('encoder.after_norm.bias'), eps=1e-12)
        return x.numpy()

    # ---- CFM estimator (CausalConditionalDecoder) ----
    def _e(self, k): return self.F['decoder.estimator.' + k]
    def _sinpos(self, t, dim=320, scale=1000):
        half = dim // 2; emb = math.log(10000) / (half - 1)
        emb = np.exp(np.arange(half) * -emb)
        emb = scale * t[:, None].astype(np.float32) * emb[None, :].astype(np.float32)
        return np.concatenate([np.sin(emb), np.cos(emb)], -1).astype(np.float32)
    def _elin(self, x, p, bias=True):
        y = x @ self._e(p + '.weight').T
        if bias and ('decoder.estimator.' + p + '.bias') in self.F: y = y + self._e(p + '.bias')
        return y
    def _cblock1d(self, x, mask, p):
        xm = x * mask
        h = _causal_conv1d(xm, self._e(p + '.block.0.weight'), self._e(p + '.block.0.bias'))
        h = h.transpose(0, 2, 1)
        h = _layernorm(h, self._e(p + '.block.2.weight'), self._e(p + '.block.2.bias'))
        h = h.transpose(0, 2, 1)
        return _mish(h) * mask
    def _cresnet(self, x, mask, t, p):
        h = self._cblock1d(x, mask, p + '.block1')
        temb = _mish(t) @ self._e(p + '.mlp.1.weight').T + self._e(p + '.mlp.1.bias')
        h = h + temb[:, :, None]
        h = self._cblock1d(h, mask, p + '.block2')
        res = _conv1d(x * mask, self._e(p + '.res_conv.weight'), self._e(p + '.res_conv.bias'))
        return h + res
    def _eattn(self, x, p, heads=8, dh=64):
        B, S, C = x.shape
        q = self._elin(x, p + '.to_q', bias=False); k = self._elin(x, p + '.to_k', bias=False); v = self._elin(x, p + '.to_v', bias=False)
        def sp(z): return z.reshape(B, S, heads, dh).transpose(0, 2, 1, 3)
        q, k, v = sp(q), sp(k), sp(v); scale = dh ** -0.5
        a = (q * scale) @ k.transpose(0, 1, 3, 2)
        a = a - a.max(-1, keepdims=True); a = np.exp(a); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, S, heads * dh)
        return self._elin(o, p + '.to_out.0')
    def _etblock(self, x, p):
        n = _layernorm(x, self._e(p + '.norm1.weight'), self._e(p + '.norm1.bias'))
        x = x + self._eattn(n, p + '.attn1')
        n = _layernorm(x, self._e(p + '.norm3.weight'), self._e(p + '.norm3.bias'))
        h = self._elin(n, p + '.ff.net.0.proj'); h = _gelu(h); h = self._elin(h, p + '.ff.net.2')
        return x + h
    def _etransformers(self, x_bct, base, n=4):
        x = x_bct.transpose(0, 2, 1)
        for i in range(n): x = self._etblock(x, f'{base}.{i}')
        return x.transpose(0, 2, 1)
    def _estimator(self, x, mask, mu, t, spks, cond):
        B, _, T = x.shape
        temb = self._sinpos(t)
        temb = _silu(temb @ self._e('time_mlp.linear_1.weight').T + self._e('time_mlp.linear_1.bias'))
        temb = temb @ self._e('time_mlp.linear_2.weight').T + self._e('time_mlp.linear_2.bias')
        x = np.concatenate([x, mu], 1)
        x = np.concatenate([x, np.repeat(spks[:, :, None], T, axis=2)], 1)
        x = np.concatenate([x, cond], 1)
        hiddens = []; masks = [mask]; md = masks[-1]
        x = self._cresnet(x, md, temb, 'down_blocks.0.0'); x = self._etransformers(x, 'down_blocks.0.1')
        hiddens.append(x)
        x = _causal_conv1d(x * md, self._e('down_blocks.0.2.weight'), self._e('down_blocks.0.2.bias'))
        masks.append(md[:, :, ::2]); masks = masks[:-1]; mm = masks[-1]
        for i in range(12):
            x = self._cresnet(x, mm, temb, f'mid_blocks.{i}.0'); x = self._etransformers(x, f'mid_blocks.{i}.1')
        mu_up = masks.pop(); skip = hiddens.pop()
        x = np.concatenate([x[:, :, :skip.shape[-1]], skip], 1)
        x = self._cresnet(x, mu_up, temb, 'up_blocks.0.0'); x = self._etransformers(x, 'up_blocks.0.1')
        x = _causal_conv1d(x * mu_up, self._e('up_blocks.0.2.weight'), self._e('up_blocks.0.2.bias'))
        x = self._cblock1d(x, mu_up, 'final_block')
        out = _conv1d(x * mu_up, self._e('final_proj.weight'), self._e('final_proj.bias'))
        return out * mask

    # ============ GPU estimator (webtorch Tensors, graph-capturable) ============
    def _gpu_ok(self):
        return wt is not None and getattr(wt, 'GPU', False)

    def _wtT(self, a):
        return wt.Tensor(np.ascontiguousarray(a.astype(np.float32)))

    def _build_ew(self):
        # cache ALL estimator weights as persistent webtorch Tensors (needed so a
        # captured graph's kernels keep referencing stable buffers on replay).
        if getattr(self, '_ew', None) is not None: return
        self._ew = {}
        for k in self.F:
            if not k.startswith('decoder.estimator.'): continue
            kk = k[len('decoder.estimator.'):]; a = self.F[k]
            if kk.endswith('.weight') and a.ndim == 2:          # linear -> pre-transpose (I,O)
                self._ew[kk] = self._wtT(a.T)
            elif kk.endswith('.weight') and a.ndim == 3:        # conv (O,C,K) -> per-tap (C,O)
                O, C, K = a.shape
                for t in range(K): self._ew[f'{kk}.tap{t}'] = self._wtT(a[:, :, t].T)
                self._ew[f'{kk}.O'] = O
            else:
                self._ew[kk] = self._wtT(a)

    def _wt_conv(self, x, p, ksz, causal, bias=True):
        # GPU-safe conv1d via matmul accumulation (WgPy has no xp.stack). x (B,C,T).
        B, C, T = x.shape
        xp = wt.cat([wt.Tensor(np.zeros((B, C, ksz - 1), np.float32)), x], axis=-1) if causal else x
        O = self._ew[f'{p}.weight.O']; out = None
        for t in range(ksz):
            xs = wt._slice_last(xp, t, t + T)                   # (B,C,T)
            y = xs.permute(0, 2, 1).reshape(B * T, C).matmul(self._ew[f'{p}.weight.tap{t}'])
            out = y if out is None else out + y
        out = out.reshape(B, T, O).permute(0, 2, 1)             # (B,O,T)
        b = self._ew.get(p + '.bias') if bias else None
        if b is not None: out = out + b.reshape(1, O, 1)
        return out

    def _wt_lin(self, x, p, bias=True):
        w = self._ew[p + '.weight']; b = self._ew.get(p + '.bias') if bias else None
        sh = x.shape
        x2 = x.reshape(int(np.prod(sh[:-1])), sh[-1]) if x.ndim > 2 else x
        y = x2.matmul(w)
        if b is not None: y = y + b
        return y.reshape(*sh[:-1], y.shape[-1]) if x.ndim > 2 else y

    @staticmethod
    def _wt_mish(x):
        ax = x.abs(); sp = (x + ax) * 0.5 + (((ax * (-1.0)).exp()) + 1.0).log()
        return x * sp.tanh()

    @staticmethod
    def _wt_gelu(x):
        y = x * (1.0 / math.sqrt(2.0)); ax = y.abs(); t = 1.0 / (1.0 + 0.3275911 * ax)
        poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
        erf = (1.0 - poly * ((y * y) * (-1.0)).exp()) * (y * (1.0 / (ax + 1e-12)))
        return x * 0.5 * (1.0 + erf)

    def _wt_cblock(self, x, p):
        h = self._wt_conv(x, p + '.block.0', 3, True).permute(0, 2, 1)
        h = wt.layernorm(h, self._ew[p + '.block.2.weight'], self._ew[p + '.block.2.bias'])
        return self._wt_mish(h.permute(0, 2, 1))

    def _wt_cresnet(self, x, temb, p):
        h = self._wt_cblock(x, p + '.block1')
        tt = self._wt_lin(self._wt_mish(temb), p + '.mlp.1')
        h = h + tt.reshape(tt.shape[0], tt.shape[1], 1)
        h = self._wt_cblock(h, p + '.block2')
        res = self._wt_conv(x, p + '.res_conv', 1, False)
        return h + res

    def _wt_attn(self, x, p, heads=8, dh=64):
        B, S, C = x.shape
        q = self._wt_lin(x, p + '.to_q', False); k = self._wt_lin(x, p + '.to_k', False); v = self._wt_lin(x, p + '.to_v', False)
        def sp(z): return z.reshape(B, S, heads, dh).permute(0, 2, 1, 3).reshape(B * heads, S, dh)
        q, k, v = sp(q), sp(k), sp(v)
        a = wt.softmax(wt.bmm(q * (dh ** -0.5), k.permute(0, 2, 1)))
        o = wt.bmm(a, v).reshape(B, heads, S, dh).permute(0, 2, 1, 3).reshape(B, S, heads * dh)
        return self._wt_lin(o, p + '.to_out.0')

    def _wt_tblock(self, x, p):
        n = wt.layernorm(x, self._ew[p + '.norm1.weight'], self._ew[p + '.norm1.bias'])
        x = x + self._wt_attn(n, p + '.attn1')
        n = wt.layernorm(x, self._ew[p + '.norm3.weight'], self._ew[p + '.norm3.bias'])
        h = self._wt_lin(n, p + '.ff.net.0.proj'); h = self._wt_gelu(h); h = self._wt_lin(h, p + '.ff.net.2')
        return x + h

    def _wt_transformers(self, x_bct, base, n=4):
        x = x_bct.permute(0, 2, 1)
        for i in range(n): x = self._wt_tblock(x, f'{base}.{i}')
        return x.permute(0, 2, 1)

    def _estimator_wt(self, x, temb, rest):
        xx = wt.cat([x, rest], axis=1)
        xx = self._wt_cresnet(xx, temb, 'down_blocks.0.0'); xx = self._wt_transformers(xx, 'down_blocks.0.1')
        hidden = xx
        xx = self._wt_conv(xx, 'down_blocks.0.2', 3, True)
        for i in range(12):
            xx = self._wt_cresnet(xx, temb, f'mid_blocks.{i}.0'); xx = self._wt_transformers(xx, f'mid_blocks.{i}.1')
        xx = wt.cat([xx, hidden], axis=1)
        xx = self._wt_cresnet(xx, temb, 'up_blocks.0.0'); xx = self._wt_transformers(xx, 'up_blocks.0.1')
        xx = self._wt_conv(xx, 'up_blocks.0.2', 3, True)
        xx = self._wt_cblock(xx, 'final_block')
        return self._wt_conv(xx, 'final_proj', 1, False)

    def _temb(self, t):
        e = self._sinpos(np.array([t, t], np.float32))
        e = _silu(e @ self._e('time_mlp.linear_1.weight').T + self._e('time_mlp.linear_1.bias'))
        return (e @ self._e('time_mlp.linear_2.weight').T + self._e('time_mlp.linear_2.bias')).astype(np.float32)

    def _cfm_gpu(self, mu, spks, cond, n_timesteps=10, cfg=0.7, capture=True):
        self._build_ew()
        T = mu.shape[2]
        z = self.baked['cfm_z'][:, :, :T].astype(np.float32) if ('cfm_z' in self.baked and self.baked['cfm_z'].shape[2] >= T) \
            else np.random.RandomState(0).randn(1, 80, T).astype(np.float32)
        muin = np.concatenate([mu, np.zeros_like(mu)], 0); spin = np.concatenate([spks, np.zeros_like(spks)], 0)
        coin = np.concatenate([cond, np.zeros_like(cond)], 0)
        rest = self._wtT(np.concatenate([muin, np.repeat(spin[:, :, None], T, axis=2), coin], 1))
        t_span = 1 - np.cos(np.linspace(0, 1, n_timesteps + 1) * 0.5 * np.pi)
        x = z.copy(); tcur = t_span[0]; dt = t_span[1] - t_span[0]

        use_cap = capture and wt is not None and wt._adam_backend_ready()
        if use_cap:
            x_in = wt.Tensor(np.concatenate([x, x], 0).astype(np.float32))
            temb_in = wt.Tensor(self._temb(tcur))
            plat = wt._adam_kernel["platform"]
            plat.beginCapture("cfm_est")
            out_t = self._estimator_wt(x_in, temb_in, rest); out_t.numpy()
            plat.endCapture()
            for step in range(1, len(t_span)):
                x_in.data.buffer.set_data(np.concatenate([x, x], 0).astype(np.float32).reshape(-1))
                temb_in.data.buffer.set_data(self._temb(tcur).reshape(-1))
                plat.replay("cfm_est")
                dphi = out_t.numpy()
                x = (x + np.float32(dt) * ((1.0 + cfg) * dphi[:1] - cfg * dphi[1:])).astype(np.float32)
                tcur += dt
                if step < len(t_span) - 1: dt = t_span[step + 1] - tcur
        else:                                   # per-op GPU (WebGL) or numpy fallback
            for step in range(1, len(t_span)):
                xin = wt.Tensor(np.concatenate([x, x], 0).astype(np.float32))
                dphi = self._estimator_wt(xin, wt.Tensor(self._temb(tcur)), rest).numpy()
                x = (x + np.float32(dt) * ((1.0 + cfg) * dphi[:1] - cfg * dphi[1:])).astype(np.float32)
                tcur += dt
                if step < len(t_span) - 1: dt = t_span[step + 1] - tcur
        return x

    def _cfm(self, mu, spks, cond, n_timesteps=10, cfg=0.7):
        if self._gpu_ok():
            return self._cfm_gpu(mu, spks, cond, n_timesteps, cfg)
        T = mu.shape[2]
        # native uses a seeded rand_noise[:, :, :T]; use the baked copy for exact match if it fits
        if 'cfm_z' in self.baked and self.baked['cfm_z'].shape[2] >= T:
            z = self.baked['cfm_z'][:, :, :T].astype(np.float32)
        else:
            z = np.random.RandomState(0).randn(1, 80, T).astype(np.float32)
        t_span = 1 - np.cos(np.linspace(0, 1, n_timesteps + 1) * 0.5 * np.pi)
        x = z.copy(); t = t_span[0]; dt = t_span[1] - t_span[0]
        mask = np.ones((1, 1, T), np.float32)
        for step in range(1, len(t_span)):
            x_in = np.concatenate([x, x], 0); mask_in = np.concatenate([mask, mask], 0)
            mu_in = np.concatenate([mu, np.zeros_like(mu)], 0)
            t_in = np.array([t, t], np.float32)
            spks_in = np.concatenate([spks, np.zeros_like(spks)], 0)
            cond_in = np.concatenate([cond, np.zeros_like(cond)], 0)
            dphi = self._estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            c, u = dphi[:1], dphi[1:]
            dphi = ((1.0 + cfg) * c - cfg * u)
            x = (x + np.float32(dt) * dphi).astype(np.float32); t = t + dt
            if step < len(t_span) - 1: dt = t_span[step + 1] - t
        return x

    def flow(self, token, prompt_token, prompt_feat, embedding, n_timesteps=10):  # noqa
        full = np.concatenate([prompt_token, token]).astype(np.int64)
        emb = self.F['input_embedding.weight'][np.clip(full, 0, None)]
        h = self._encoder_gpu(emb) if self._gpu_ok() else self._encoder(emb)
        mu = self._lin(h, 'encoder_proj').T[None].astype(np.float32)
        T = mu.shape[2]; mel_len1 = prompt_feat.shape[0]
        cond = np.zeros((1, 80, T), np.float32); cond[0, :, :mel_len1] = prompt_feat.T
        e = embedding / (np.linalg.norm(embedding) + 1e-12)
        spks = (e[None] @ self.F['spk_embed_affine_layer.weight'].T + self.F['spk_embed_affine_layer.bias']).astype(np.float32)
        feat = self._cfm(mu, spks, cond, n_timesteps)
        return feat[:, :, mel_len1:]

    # ================= VOCODER (HiFTGenerator) =================
    def _hw(self, p):
        H = self.H
        if p + '.parametrizations.weight.original0' in H:
            g = H[p + '.parametrizations.weight.original0']; v = H[p + '.parametrizations.weight.original1']
            w = g * v / np.sqrt((v ** 2).sum((1, 2), keepdims=True))
        elif p + '.weight_g' in H:
            g = H[p + '.weight_g']; v = H[p + '.weight_v']; w = g * v / np.sqrt((v ** 2).sum((1, 2), keepdims=True))
        else:
            w = H[p + '.weight']
        b = H[p + '.bias'] if p + '.bias' in H else None
        return w.astype(np.float32), (b.astype(np.float32) if b is not None else None)
    def _f0(self, mel):
        x = mel
        for i in [0, 2, 4, 6, 8]:
            x = _elu(_conv1d(x, *self._hw(f'f0_predictor.condnet.{i}'), pad=1))
        x = x.T
        cl = self.H['f0_predictor.classifier.weight']; cb = self.H['f0_predictor.classifier.bias']
        return np.abs((x @ cl.T + cb)[:, 0])
    def _interp_lin(self, x, scale):
        C, L = x.shape; Lo = int(round(L * scale))
        if Lo == L: return x.copy()
        idx = (np.arange(Lo) + 0.5) / scale - 0.5
        idx = np.clip(idx, 0, L - 1); lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, L - 1); fr = idx - lo
        return x[:, lo] * (1 - fr) + x[:, hi] * fr
    def _nsf_source(self, f0, seed=0):
        rng = np.random.RandomState(seed)
        up = 480; harm = 8; sr = 24000; sine_amp = 0.1; noise_std = 0.003
        f0u = np.repeat(f0, up)
        fn = f0u[:, None] * np.arange(1, harm + 2)[None, :]
        rad = (fn / sr) % 1.0
        rand_ini = rng.rand(harm + 1).astype(np.float32); rand_ini[0] = 0
        rad[0, :] = rad[0, :] + rand_ini
        rad_ds = self._interp_lin(rad.T, 1.0 / up)
        phase = np.cumsum(rad_ds, axis=1) * 2 * np.pi
        phase_up = self._interp_lin(phase * up, up)
        Lp = phase_up.shape[1]
        sines = np.sin(phase_up).T * sine_amp
        uv = (f0u > 0).astype(np.float32)[:Lp, None]
        noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
        noise = noise_amp * rng.randn(Lp, harm + 1).astype(np.float32)
        sine_waves = sines * uv + noise
        ll_w = self.baked['ll_w']; ll_b = self.baked['ll_b']
        return np.tanh(sine_waves @ ll_w.T + ll_b)[:, 0]
    def _stft(self, x, NFFT=16, HOP=4):
        win = _hann(NFFT); xp = np.pad(x, NFFT // 2, mode='reflect'); TT = 1 + (len(xp) - NFFT) // HOP
        fr = np.stack([xp[t * HOP:t * HOP + NFFT] * win for t in range(TT)], 0)
        F = np.fft.rfft(fr, axis=1).T
        return F.real.astype(np.float32), F.imag.astype(np.float32)
    def _istft(self, mag, phase, length, NFFT=16, HOP=4):
        win = _hann(NFFT); spec = mag * np.cos(phase) + 1j * mag * np.sin(phase)
        TT = spec.shape[1]; fr = np.fft.irfft(spec, n=NFFT, axis=0) * win[:, None]
        Ltot = NFFT + HOP * (TT - 1); out = np.zeros(Ltot, np.float32); wsum = np.zeros(Ltot, np.float32)
        for t in range(TT):
            out[t * HOP:t * HOP + NFFT] += fr[:, t]; wsum[t * HOP:t * HOP + NFFT] += win ** 2
        out = out / (wsum + 1e-11); return out[NFFT // 2:NFFT // 2 + length]
    def _snake(self, x, a): a = a[:, None]; return x + (1.0 / (a + 1e-9)) * np.sin(a * x) ** 2
    def _resblock(self, x, pfx, dils):
        for j, d in enumerate(dils):
            a1 = self.H[f'{pfx}.activations1.{j}.alpha']; a2 = self.H[f'{pfx}.activations2.{j}.alpha']
            xt = self._snake(x, a1); k = self._hw(f'{pfx}.convs1.{j}')[0].shape[2]
            xt = _conv1d(xt, *self._hw(f'{pfx}.convs1.{j}'), pad=(k * d - d) // 2, dil=d)
            xt = self._snake(xt, a2); xt = _conv1d(xt, *self._hw(f'{pfx}.convs2.{j}'), pad=(k - 1) // 2)
            x = xt + x
        return x
    def _decode(self, mel, src):
        UPS = [(8, 16, 4), (5, 11, 3), (3, 7, 2)]; RK = [[1, 3, 5]] * 3; SRD = [[1, 3, 5]] * 3
        sr, si = self._stft(src); s_stft = np.concatenate([sr, si], 0)
        x = _conv1d(mel, *self._hw('conv_pre'), pad=3)
        for i in range(3):
            x = _lrelu(x, 0.1); u, k, p = UPS[i]
            x = _convT1d(x, *self._hw(f'ups.{i}'), u, p)
            if i == 2: x = np.pad(x, ((0, 0), (1, 0)), mode='reflect')
            sd = _conv1d(s_stft, *self._hw(f'source_downs.{i}'), s=(15 if i == 0 else 3 if i == 1 else 1), pad=(7 if i == 0 else 1 if i == 1 else 0))
            sd = self._resblock(sd, f'source_resblocks.{i}', SRD[i])
            x = x + sd
            xs = None
            for j in range(3):
                rb = self._resblock(x, f'resblocks.{i * 3 + j}', RK[j]); xs = rb if xs is None else xs + rb
            x = xs / 3.0
        x = _lrelu(x, 0.01); x = _conv1d(x, *self._hw('conv_post'), pad=3)
        mag = np.exp(np.clip(x[:9], None, 1e2)); ph = np.sin(x[9:])
        return np.clip(self._istft(mag, ph, len(src)), -0.99, 0.99)

    # ---- GPU vocoder decode (xp arrays; conv=matmul-accum, convT=zero-insert, snake=sin kernel) ----
    def _hxp_conv(self, p, convT=False):
        c = self.__dict__.setdefault('_hxpc', {}); key = (p, convT)
        if key in c: return c[key]
        w, b = self._hw(p); K = w.shape[2]; xp = wt.xp
        if convT:  # weight (Cin,Cout,K) -> per-tap (Cout,Cin)
            taps = [xp.asarray(np.ascontiguousarray(w[:, :, k].T.astype(np.float32))) for k in range(K)]
        else:      # weight (O,C,K) -> per-tap (O,C)
            taps = [xp.asarray(np.ascontiguousarray(w[:, :, k].astype(np.float32))) for k in range(K)]
        bx = xp.asarray(b.astype(np.float32)) if b is not None else None
        c[key] = (taps, bx); return c[key]

    def _hxp_alpha(self, k):
        c = self.__dict__.setdefault('_hxpa', {})
        if k not in c: c[k] = wt.xp.asarray(self.H[k].astype(np.float32))
        return c[k]

    def _gconv(self, x, p, pad=0, dil=1, s=1):
        xp = wt.xp; taps, b = self._hxp_conv(p); C = x.shape[0]; K = len(taps)
        if pad:
            xpad = xp.zeros((C, x.shape[1] + 2 * pad), np.float32); xpad[:, pad:pad + x.shape[1]] = x
        else:
            xpad = x
        Lp = xpad.shape[1]; eff = (K - 1) * dil + 1; Lo = (Lp - eff) // s + 1
        out = None
        for k in range(K):
            xs = xpad[:, k * dil:k * dil + s * Lo:s]
            y = taps[k] @ xs
            out = y if out is None else out + y
        if b is not None: out = out + b[:, None]
        return out

    def _gconvT(self, x, p, stride, pad):
        xp = wt.xp; taps, b = self._hxp_conv(p, convT=True)
        K = len(taps); L = x.shape[1]; O = taps[0].shape[0]; Lfull = (L - 1) * stride + K
        out = xp.zeros((O, Lfull), np.float32)
        for k in range(K):
            tmp = xp.zeros((O, Lfull), np.float32); tmp[:, k:k + stride * L:stride] = taps[k] @ x
            out = out + tmp
        if b is not None: out = out + b[:, None]
        return out[:, pad:Lfull - pad]

    def _gsnake(self, x, akey):
        a = self._hxp_alpha(akey)[:, None]; s = wt._sin_data(a * x)
        return x + (1.0 / (a + 1e-9)) * (s * s)

    @staticmethod
    def _glrelu(x, s=0.1):
        xp = wt.xp; return xp.clip(x, 0.0, 1e30) - s * xp.clip(x * (-1.0), 0.0, 1e30)

    def _gresblock(self, x, pfx, dils):
        for j, d in enumerate(dils):
            k = self._hw(f'{pfx}.convs1.{j}')[0].shape[2]
            xt = self._gsnake(x, f'{pfx}.activations1.{j}.alpha')
            xt = self._gconv(xt, f'{pfx}.convs1.{j}', pad=(k * d - d) // 2, dil=d)
            xt = self._gsnake(xt, f'{pfx}.activations2.{j}.alpha')
            xt = self._gconv(xt, f'{pfx}.convs2.{j}', pad=(k - 1) // 2)
            x = xt + x
        return x

    def _decode_gpu(self, mel, src):
        xp = wt.xp
        UPS = [(8, 16, 4), (5, 11, 3), (3, 7, 2)]; RK = [[1, 3, 5]] * 3; SRD = [[1, 3, 5]] * 3
        sr, si = self._stft(src); s_stft = xp.asarray(np.concatenate([sr, si], 0).astype(np.float32))
        x = self._gconv(xp.asarray(mel.astype(np.float32)), 'conv_pre', pad=3)
        for i in range(3):
            x = self._glrelu(x, 0.1); u, k, p = UPS[i]
            x = self._gconvT(x, f'ups.{i}', u, p)
            if i == 2:
                xpad = xp.zeros((x.shape[0], x.shape[1] + 1), np.float32); xpad[:, 1:] = x; xpad[:, 0:1] = x[:, 1:2]
                x = xpad
            sd = self._gconv(s_stft, f'source_downs.{i}', s=(15 if i == 0 else 3 if i == 1 else 1), pad=(7 if i == 0 else 1 if i == 1 else 0))
            sd = self._gresblock(sd, f'source_resblocks.{i}', SRD[i])
            x = x + sd
            xs = None
            for j in range(3):
                rb = self._gresblock(x, f'resblocks.{i * 3 + j}', RK[j]); xs = rb if xs is None else xs + rb
            x = xs * (1.0 / 3.0)
        x = self._glrelu(x, 0.01); x = self._gconv(x, 'conv_post', pad=3)
        xnp = wt.xp.asnumpy(x) if wt.GPU else np.asarray(x)
        mag = np.exp(np.clip(xnp[:9], None, 1e2)); ph = np.sin(xnp[9:])
        return np.clip(self._istft(mag, ph, len(src)), -0.99, 0.99)

    def vocoder(self, mel, seed=0):
        f0 = self._f0(mel)
        src = self._nsf_source(f0, seed)
        if self._gpu_ok():
            return self._decode_gpu(mel, src)
        return self._decode(mel, src)

    # ================= LLM (text -> speech tokens), via generic lm_engine =================
    async def load_llm(self, model="/models/cosy_llm.npz", tokenizer="/models/cosy_qwen_tok.json", io=None):
        from . import lm_engine, webio
        self.LZ = await webio.load_npz(model, io)           # url | bytes | callback
        cfg = json.loads(bytes(self.LZ['config']).decode())
        lm = lm_engine.TransformerLM(cfg)
        def ql(pfx):
            K, N, Kp, Np = [int(v) for v in self.LZ[pfx + '.dims']]
            b = self.LZ[pfx + '.bias'] if pfx + '.bias' in self.LZ else np.zeros((N,), np.float32)
            return wt.QuantizedLinear(self.LZ[pfx + '.qweight'], self.LZ[pfx + '.qzeros'],
                                      self.LZ[pfx + '.scales'], b, K, N, Kp, Np, cfg['gs'], cfg['bits'])
        for i in range(cfg['L']):
            p = f'layers.{i}.'
            lm.layers.append({
                'in_ln': wt.Tensor(self.LZ[p + 'in_ln']), 'post_ln': wt.Tensor(self.LZ[p + 'post_ln']),
                'q': ql(p + 'q'), 'k': ql(p + 'k'), 'v': ql(p + 'v'), 'o': ql(p + 'o'),
                'gate': ql(p + 'gate'), 'up': ql(p + 'up'), 'down': ql(p + 'down')})
        lm.final_norm = wt.Tensor(self.LZ['final_norm'])
        head = ql('llm_decoder')
        lm.head = lambda hlast: head(hlast)
        self._speech_emb = self.LZ['speech_embedding'].astype(np.float32)     # (6564,896)
        self._embed_tokens = self.LZ['embed_tokens']                          # (151936,896) f16
        self._llm_emb = self.LZ['llm_embedding'].astype(np.float32)           # (2,896) sos/task
        lm.embed_next = lambda tid: self._speech_emb[tid]
        self.lm = lm; self.speech_vocab = cfg['speech_vocab']
        self._llm_captured = lm.init_capture(lmax=512)          # WebGPU capture-replay decode (~20x)
        # Qwen BPE tokenizer (generic, from served vocab/merges json)
        from . import llm as _llm
        tj = json.loads(bytes(await webio.read_bytes(tokenizer, io)).decode())
        self.tok = _llm.BPETokenizer(tj['vocab'], tj['merges'])
        return self

    def _assemble_prompt(self, text_ids, prompt_speech_token):
        # [sos ; text_emb ; task_id ; prompt_speech_emb]
        sos = self._llm_emb[0:1]; task = self._llm_emb[1:2]
        te = self._embed_tokens[np.asarray(text_ids, np.int64)].astype(np.float32)
        pe = self._speech_emb[np.asarray(prompt_speech_token, np.int64)]
        return np.concatenate([sos, te, task, pe], 0).astype(np.float32)

    def generate_speech_tokens(self, text, prompt_text=None, prompt_speech_token=None, seed=0, max_ratio=20, min_ratio=2):
        # prompt_text MUST match the prompt speaker's audio (else generation is incoherent);
        # default to the baked speaker's transcript.
        if prompt_text is None:
            prompt_text = str(self.baked['prompt_text']) if 'prompt_text' in self.baked else ""
        ids = self.tok.encode(prompt_text + text) if prompt_text else self.tok.encode(text)
        if prompt_speech_token is None:
            prompt_speech_token = self.baked['flow_prompt_token'][0]
        embs = wt.Tensor(self._assemble_prompt(ids, prompt_speech_token))
        n_text = len(self.tok.encode(text))
        stop = set(range(self.speech_vocab, self.speech_vocab + 3))   # {6561,6562,6563}
        gen = self.lm.generate_captured if getattr(self, '_llm_captured', False) else self.lm.generate
        toks = gen(embs, max_new=max(4, int(n_text * max_ratio)), stop_ids=stop,
                   sampler="ras", sampler_kwargs=dict(top_p=0.8, top_k=25, win_size=10, tau_r=0.1),
                   seed=seed, min_new=max(4, int(n_text * min_ratio)))
        return np.array(toks, np.int64)

    # ---- zero-shot voice cloning: prompt audio -> features (generic ONNX + audiofe) ----
    async def load_clone(self, spk_tok_onnx="/models/speech_tokenizer_v2.onnx",
                         campplus_onnx="/models/campplus.onnx", melfilters="/models/cosy_melfilters.npz", io=None):
        from . import onnxrt, webio                                    # each arg: url | bytes | callback
        self._spk_tok = await onnxrt.OnnxModel.from_source(spk_tok_onnx, io)
        self._campplus = await onnxrt.OnnxModel.from_source(campplus_onnx, io)
        self._mf = await webio.load_npz(melfilters, io)
        self.can_clone = True                         # cloning now available (generic protocol flag)
        return self

    def prompt_features(self, wav16, wav24):
        """prompt waveforms (16k, 24k) -> (prompt_speech_token, spk_embedding, prompt_feat)."""
        from . import audiofe
        wm = audiofe.whisper_log_mel(np.asarray(wav16, np.float32), self._mf['whisper'])
        tok = self._spk_tok.run({'feats': wm[None].astype(np.float32),
                                 'feats_length': np.array([wm.shape[1]], np.int32)})[0].ravel().astype(np.int64)
        fb = audiofe.kaldi_fbank(np.asarray(wav16, np.float32), self._mf['kaldi'])
        emb = self._campplus.run({'input': fb[None].astype(np.float32)})[0].ravel().astype(np.float32)
        feat = audiofe.mel_spectrogram(np.asarray(wav24, np.float32), self._mf['matcha']).T.astype(np.float32)
        return tok, emb, feat

    def inference_zero_shot(self, text, prompt_text, wav16, wav24, seed=0):
        """Full zero-shot voice cloning (mirrors native CosyVoice2 API)."""
        tok, emb, feat = self.prompt_features(wav16, wav24)
        st = self.generate_speech_tokens(text, prompt_text, tok, seed)
        mel = self.flow(st, tok, feat, emb)
        wav = self.vocoder(mel[0], seed)
        return wav, mel, st

    def tts(self, text, prompt_text=None, prompt_speech_token=None, prompt_feat=None, embedding=None, seed=0):
        """Full text -> waveform. Uses baked speaker features unless overridden."""
        b = self.baked
        st = self.generate_speech_tokens(text, prompt_text, prompt_speech_token, seed)
        mel = self.flow(st,
                        prompt_speech_token if prompt_speech_token is not None else b['flow_prompt_token'][0],
                        prompt_feat if prompt_feat is not None else b['flow_prompt_feat'][0],
                        embedding if embedding is not None else b['flow_embedding'][0])
        wav = self.vocoder(mel[0], seed)
        return wav, mel, st

    # ---- generic TTS protocol (consumed by the pipeline; no model-specific branch upstream) ----
    can_clone = False                                # set True once load_clone() has run
    def synth(self, text, **kw):
        return self.tts(text, **kw)[0]
    def clone(self, text, reference_audio, reference_text="", **kw):
        w16, w24 = reference_audio
        return self.inference_zero_shot(text, reference_text, w16, w24, **kw)[0]

    # ================= end-to-end =================
    def synthesize(self, token=None, seed=0, n_timesteps=10):
        b = self.baked
        if token is None: token = b['flow_token'][0]
        mel = self.flow(token, b['flow_prompt_token'][0], b['flow_prompt_feat'][0], b['flow_embedding'][0], n_timesteps)
        wav = self.vocoder(mel[0], seed)
        return wav, mel
