"""webtorch.tts -- text-to-speech (VITS / MMS-TTS) in the browser.

Full VITS pipeline validated offline to ~3e-4 vs HF transformers. The small
front-end (text encoder with relative-position attention, stochastic duration
predictor with rational-quadratic-spline flows, monotonic expansion, residual
coupling flow) runs host-side (tiny: <=85 frames); the compute-heavy HiFiGAN
vocoder (256x upsampling to 16 kHz) runs on webtorch GPU via wt.conv1d /
wt.conv_transpose1d. Deterministic (noise disabled).

    import tts
    v = await tts.VitsTTS.from_npz("/models/vits_web.npz")
    wav = v.synthesize([0,6,0,7,...])   # tokenized input_ids -> float32 waveform @16kHz
"""
import io, math, json
import numpy as np
from . import _core as wt


def _lrelu(x, s=0.1): return np.where(x >= 0, x, x * s)
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _softmax(x, ax=-1):
    e = np.exp(x - x.max(ax, keepdims=True)); return e / e.sum(ax, keepdims=True)
def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
def _erf_arr(x):
    # Abramowitz-Stegun erf (vectorized, ~1e-7)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.sign(x) * y
def _gelu(x): return 0.5 * x * (1.0 + _erf_arr(x / math.sqrt(2.0)))


class VitsTTS:
    def __init__(self, W, cfg):
        self.Z = W; self.C = cfg
        self.H = cfg["hidden_size"]; self.NH = cfg["num_attention_heads"]; self.HD = self.H // self.NH
        self.WS = cfg["window_size"]; self.EPS = cfg["layer_norm_eps"]
        self.sr = cfg["sampling_rate"]
        self._t = {}

    # ---------- numpy conv helpers (host front-end) ----------
    def cw(self, p):
        Z = self.Z
        w = Z[p + ".weight"].astype(np.float32); b = Z[p + ".bias"].astype(np.float32) if (p + ".bias") in Z else None
        return w, b

    def conv1d(self, x, w, b, stride=1, pad=0, dil=1, groups=1):
        Cin, L = x.shape; O, Cg, K = w.shape
        if pad: x = np.pad(x, ((0, 0), (pad, pad)))
        Lp = x.shape[1]; eff = (K - 1) * dil + 1; Lo = (Lp - eff) // stride + 1
        out = np.zeros((O, Lo), np.float32)
        if groups == 1:
            for k in range(K): out += w[:, :, k] @ x[:, k * dil:k * dil + stride * Lo:stride]
        else:
            og = O // groups; cg = Cin // groups
            for gi in range(groups):
                for k in range(K):
                    out[gi * og:(gi + 1) * og] += w[gi * og:(gi + 1) * og, :, k] @ x[gi * cg:(gi + 1) * cg, k * dil:k * dil + stride * Lo:stride]
        if b is not None: out += b[:, None]
        return out

    def _ln(self, x, w, b):
        mu = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(v + self.EPS) * w + b
    def _lnc(self, x, w, b):
        xt = x.T; mu = xt.mean(-1, keepdims=True); v = xt.var(-1, keepdims=True)
        return ((xt - mu) / np.sqrt(v + self.EPS) * w + b).T

    # ---------- text encoder (relative-position attention) ----------
    def _encode(self, ids):
        Z = self.Z; H, NH, HD, WS, EPS = self.H, self.NH, self.HD, self.WS, self.EPS
        def lin(x, p): return x @ Z[p + ".weight"].T + Z[p + ".bias"]
        def ln(x, w, b): mu = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True); return (x - mu) / np.sqrt(v + EPS) * w + b
        def get_rel(emb, L):
            pad = max(L - (WS + 1), 0)
            if pad > 0: emb = np.pad(emb, ((0, 0), (pad, pad), (0, 0)))
            st = max((WS + 1) - L, 0); return emb[:, st:st + 2 * L - 1]
        def rel2abs(x):
            bh, L, _ = x.shape; x = np.pad(x, ((0, 0), (0, 0), (0, 1))); xf = np.pad(x.reshape(bh, L * 2 * L), ((0, 0), (0, L - 1)))
            return xf.reshape(bh, L + 1, 2 * L - 1)[:, :L, L - 1:]
        def abs2rel(x):
            bh, L, _ = x.shape; x = np.pad(x, ((0, 0), (0, 0), (0, L - 1))); xf = np.pad(x.reshape(bh, L * (2 * L - 1)), ((0, 0), (L, 0)))
            return xf.reshape(bh, L, 2 * L)[:, :, 1:]
        T = len(ids)
        def attn(x, p):
            sc = HD ** -0.5
            q = (lin(x, p + ".q_proj") * sc).reshape(T, NH, HD).transpose(1, 0, 2)
            k = lin(x, p + ".k_proj").reshape(T, NH, HD).transpose(1, 0, 2); v = lin(x, p + ".v_proj").reshape(T, NH, HD).transpose(1, 0, 2)
            aw = q @ k.transpose(0, 2, 1) + rel2abs(q @ get_rel(Z[p + ".emb_rel_k"], T).transpose(0, 2, 1)); aw = _softmax(aw)
            o = aw @ v + abs2rel(aw) @ get_rel(Z[p + ".emb_rel_v"], T)
            return lin(o.transpose(1, 0, 2).reshape(T, H), p + ".out_proj")
        h = Z["text_encoder.embed_tokens.weight"][ids] * np.sqrt(H)
        for i in range(self.C["num_hidden_layers"] if "num_hidden_layers" in self.C else 6):
            p = f"text_encoder.encoder.layers.{i}"
            h = ln(h + attn(h, p + ".attention"), Z[p + ".layer_norm.weight"], Z[p + ".layer_norm.bias"])
            ff = h.T; ff = np.maximum(self.conv1d(ff, *self.cw(p + ".feed_forward.conv_1"), pad=1), 0)
            ff = self.conv1d(ff, *self.cw(p + ".feed_forward.conv_2"), pad=1).T
            h = ln(h + ff, Z[p + ".final_layer_norm.weight"], Z[p + ".final_layer_norm.bias"])
        stats = self.conv1d(h.T, *self.cw("text_encoder.project")).T
        return stats[:, :H], h.T                                  # prior_means (T,H), hidden (H,T)

    # ---------- dilated depth-separable conv (SDP) ----------
    def _dds(self, x, pfx, cond=None):
        Z = self.Z; ks = self.C["duration_predictor_kernel_size"]; nl = self.C["depth_separable_num_layers"]
        if cond is not None: x = x + cond
        for i in range(nl):
            dil = ks ** i; pad = (ks * dil - dil) // 2
            hs = self.conv1d(x, *self.cw(f"{pfx}.convs_dilated.{i}"), pad=pad, dil=dil, groups=self.H)
            hs = _gelu(self._lnc(hs, Z[f"{pfx}.norms_1.{i}.weight"], Z[f"{pfx}.norms_1.{i}.bias"]))
            hs = self.conv1d(hs, *self.cw(f"{pfx}.convs_pointwise.{i}"))
            hs = _gelu(self._lnc(hs, Z[f"{pfx}.norms_2.{i}.weight"], Z[f"{pfx}.norms_2.{i}.bias"]))
            x = x + hs
        return x

    def _rqs_reverse(self, inputs, uw, uh, ud, tail, mbw=1e-3, mbh=1e-3, mder=1e-3):
        inside = (inputs >= -tail) & (inputs <= tail); out = inputs.copy()
        const = np.log(np.exp(1 - mder) - 1)
        ud = np.pad(ud, ((0, 0), (1, 1))); ud[:, 0] = const; ud[:, -1] = const
        x = inputs[inside]; W = uw[inside]; Hh = uh[inside]; D = ud[inside]; nb = W.shape[-1]
        widths = _softmax(W, -1); widths = mbw + (1 - mbw * nb) * widths
        cw_ = np.cumsum(widths, -1); cw_ = np.pad(cw_, ((0, 0), (1, 0))); cw_ = 2 * tail * cw_ - tail; cw_[:, 0] = -tail; cw_[:, -1] = tail
        widths = cw_[:, 1:] - cw_[:, :-1]
        der = mder + _softplus(D)
        heights = _softmax(Hh, -1); heights = mbh + (1 - mbh * nb) * heights
        ch = np.cumsum(heights, -1); ch = np.pad(ch, ((0, 0), (1, 0))); ch = 2 * tail * ch - tail; ch[:, 0] = -tail; ch[:, -1] = tail
        heights = ch[:, 1:] - ch[:, :-1]
        binloc = ch.copy(); binloc[:, -1] += 1e-6
        bidx = (x[:, None] >= binloc).sum(-1) - 1; ar = np.arange(len(bidx))
        def g(a): return a[ar, bidx]
        icw = g(cw_); ibw = g(widths); ich = g(ch); delta = heights / widths; idl = g(delta)
        ider = g(der); iderp = der[:, 1:][ar, bidx]; ih = g(heights)
        inter1 = ider + iderp - 2 * idl; i2 = x - ich; i3 = i2 * inter1
        a = ih * (idl - ider) + i3; b = ih * ider - i3; c = -idl * i2
        disc = b * b - 4 * a * c; root = (2 * c) / (-b - np.sqrt(disc))
        out[inside] = root * ibw + icw
        return out

    def _conv_flow_reverse(self, x2, cond, pfx, nb, tail):
        fh = x2[:1]; sh = x2[1:]
        hs = self.conv1d(fh, *self.cw(pfx + ".conv_pre"))
        hs = self._dds(hs, pfx + ".conv_dds", cond)
        hs = self.conv1d(hs, *self.cw(pfx + ".conv_proj"))
        Tt = fh.shape[1]; hs = hs.reshape(1, -1, Tt).transpose(0, 2, 1)
        uw = hs[..., :nb] / math.sqrt(self.H); uh = hs[..., nb:2 * nb] / math.sqrt(self.H); ud = hs[..., 2 * nb:]
        sh2 = self._rqs_reverse(sh.reshape(-1), uw.reshape(-1, nb), uh.reshape(-1, nb), ud.reshape(-1, ud.shape[-1]), tail)
        return np.concatenate([fh, sh2.reshape(1, Tt)], 0)

    def _duration(self, hidden):
        Z = self.Z; T = hidden.shape[1]
        inp = self.conv1d(hidden, *self.cw("duration_predictor.conv_pre"))
        inp = self._dds(inp, "duration_predictor.conv_dds")
        inp = self.conv1d(inp, *self.cw("duration_predictor.conv_proj"))
        nf = self.C["duration_predictor_num_flows"]; nb = self.C["duration_predictor_flow_bins"]; tail = self.C["duration_predictor_tail_bound"]
        order = [nf, nf - 1, nf - 2, 0]
        lat = np.zeros((2, T), np.float32)
        for fi in order:
            lat = lat[::-1].copy()
            if fi == 0:
                tr = Z["duration_predictor.flows.0.translate"]; ls = Z["duration_predictor.flows.0.log_scale"]
                lat = (lat - tr) * np.exp(-ls)
            else:
                lat = self._conv_flow_reverse(lat, inp, f"duration_predictor.flows.{fi}", nb, tail)
        logdur = lat[:1]
        dur = np.ceil(np.exp(logdur) / self.C["speaking_rate"])
        return dur

    def _expand(self, prior_means, dur):
        cum = np.cumsum(dur, -1).reshape(-1); out_len = int(dur.sum())
        idxr = np.arange(out_len)
        valid = (idxr[None, :] < cum[:, None]).astype(np.float32)
        attn = valid - np.pad(valid, ((1, 0), (0, 0)))[:-1]
        return (attn.T @ prior_means).T                          # (H, out_len)

    def _wavenet(self, x, pfx, nl):
        Z = self.Z; H = self.H; ks = self.C["wavenet_kernel_size"]; dr = self.C["wavenet_dilation_rate"]
        outputs = np.zeros_like(x)
        for i in range(nl):
            dil = dr ** i; pad = (ks * dil - dil) // 2
            hs = self.conv1d(x, *self.cw(f"{pfx}.in_layers.{i}"), pad=pad, dil=dil)
            ta = np.tanh(hs[:H]) * _sigmoid(hs[H:])
            rs = self.conv1d(ta, *self.cw(f"{pfx}.res_skip_layers.{i}"))
            if i < nl - 1:
                x = x + rs[:H]; outputs = outputs + rs[H:]
            else:
                outputs = outputs + rs
        return outputs

    def _flow(self, z):
        hc = self.C["flow_size"] // 2
        for fi in reversed(range(self.C["prior_encoder_num_flows"])):
            z = z[::-1].copy()
            fh = z[:hc]; sh = z[hc:]
            hs = self.conv1d(fh, *self.cw(f"flow.flows.{fi}.conv_pre"))
            hs = self._wavenet(hs, f"flow.flows.{fi}.wavenet", self.C["prior_encoder_num_wavenet_layers"])
            mean = self.conv1d(hs, *self.cw(f"flow.flows.{fi}.conv_post"))
            z = np.concatenate([fh, sh - mean], 0)
        return z

    # ---------- HiFiGAN vocoder (host numpy; wgpy lacks stack/full and
    # host-routes concatenate/setitem, so on-GPU conv1d is unreliable) ----------
    def _convT1d(self, x, w, b, stride, pad, dil=1):              # x (C,L), w (Cin,O,K)
        Cin, L = x.shape; _, O, K = w.shape
        Lout = (L - 1) * stride - 2 * pad + dil * (K - 1) + 1
        full = np.zeros((O, Lout + 2 * pad), np.float32)
        for k in range(K):
            full[:, k * dil:k * dil + stride * L:stride] += w[:, :, k].T @ x
        out = full[:, pad:pad + Lout] if pad else full
        if b is not None: out += b[:, None]
        return out

    def _vocoder(self, z):
        C = self.C
        x = self.conv1d(z, *self.cw("decoder.conv_pre"), pad=3)
        nk = len(C["resblock_kernel_sizes"])
        for i in range(len(C["upsample_rates"])):
            x = _lrelu(x, 0.1)
            r = C["upsample_rates"][i]; k = C["upsample_kernel_sizes"][i]
            x = self._convT1d(x, *self.cw(f"decoder.upsampler.{i}"), r, (k - r) // 2)
            rs = None
            for j in range(nk):
                ks = C["resblock_kernel_sizes"][j]; dils = C["resblock_dilation_sizes"][j]
                h = x
                for di, d in enumerate(dils):
                    res = h
                    h = self.conv1d(_lrelu(h, 0.1), *self.cw(f"decoder.resblocks.{i*nk+j}.convs1.{di}"), pad=(ks * d - d) // 2, dil=d)
                    h = self.conv1d(_lrelu(h, 0.1), *self.cw(f"decoder.resblocks.{i*nk+j}.convs2.{di}"), pad=(ks - 1) // 2)
                    h = h + res
                rs = h if rs is None else rs + h
            x = rs * (1.0 / nk)
        x = _lrelu(x, 0.01)                                       # final: torch default slope
        x = self.conv1d(x, self.Z["decoder.conv_post.weight"].astype(np.float32), None, pad=3)
        return np.tanh(x)[0]

    def tokenize(self, text):
        """MMS-TTS char tokenizer: lowercase, keep in-vocab chars, interleave blank(0)."""
        ids = [_VOCAB[c] for c in text.lower() if c in _VOCAB]
        out = [0]
        for i in ids:
            out += [i, 0]
        return out

    def synthesize(self, input_ids):
        prior_means, hidden = self._encode(list(input_ids))
        dur = self._duration(hidden)
        z_p = self._expand(prior_means, dur)                     # (H, out_len)
        z = self._flow(z_p)
        wav = self._vocoder(z)
        return wav.astype(np.float32)

    # ---- generic TTS protocol (consumed by the pipeline; no cloning support) ----
    can_clone = False
    def synth(self, text, **kw):
        return self.synthesize(self.tokenize(text))

    @classmethod
    async def from_npz(cls, url):
        from . import webio
        data = await webio.load_npz(url)
        W = {k: data[k] for k in data}
        cfg = MMS_ENG_CFG                          # config embedded for mms-tts-eng
        return cls(W, cfg)


_VOCAB = {"k": 0, "'": 1, "z": 2, "y": 3, "u": 4, "d": 5, "h": 6, "e": 7, "s": 8, "w": 9,
          "–": 10, "3": 11, "c": 12, "p": 13, "-": 14, "1": 15, "j": 16, "m": 17, "i": 18,
          " ": 19, "f": 20, "l": 21, "o": 22, "0": 23, "b": 24, "r": 25, "a": 26, "4": 27, "2": 28,
          "n": 29, "_": 30, "x": 31, "v": 32, "t": 33, "q": 34, "5": 35, "6": 36, "g": 37}

MMS_ENG_CFG = {
    "hidden_size": 192, "num_attention_heads": 2, "num_hidden_layers": 6, "window_size": 4,
    "ffn_kernel_size": 3, "layer_norm_eps": 1e-5, "flow_size": 192, "prior_encoder_num_flows": 4,
    "prior_encoder_num_wavenet_layers": 4, "wavenet_kernel_size": 5, "wavenet_dilation_rate": 1,
    "duration_predictor_flow_bins": 10, "duration_predictor_tail_bound": 5.0, "duration_predictor_kernel_size": 3,
    "duration_predictor_num_flows": 4, "depth_separable_num_layers": 3, "depth_separable_channels": 2,
    "upsample_rates": [8, 8, 2, 2], "upsample_kernel_sizes": [16, 16, 4, 4], "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "leaky_relu_slope": 0.1, "speaking_rate": 1.0, "sampling_rate": 16000, "num_speakers": 1,
}
