"""Streaming quantization for the SDK — turn ANY fp16/bf16 HF model into an int4/int8
model the engine runs, without ever materializing the full fp16 model in memory.

The streaming invariant: read ONE tensor -> quantize it -> keep only the int
result -> free the fp16. Peak extra host memory = one fp16 tensor (+ the int model,
which is ~1/4..1/8 the size). Works the same whether tensors come from a local
mmap'd safetensors or a browser HTTP-Range reader.

Public:
  pack_int(W_out_in, gs, bits) -> packed int dict (qweight/qzeros/scales/dims)
  dequant_int(packed) -> fp32 (in,out)            # host-numpy execution (no GPU)
  quant_linear_factory(reader, bits, gs, mode)    # a `linear` for lm_engine.build_lm
  Quantizer.quantize_hf(reader, out_writer, ...)  # offline: fp16 HF -> served int npz
"""
import numpy as np
from . import _core as wt


def pack_int(W_out_in, gs=128, bits=4):
    """(out,in) fp16/fp32 weight -> int packed (transposed to (in,out) for matmul).
    Uses webtorch._gptq_quantize (zero-point WITHOUT the +1 AutoGPTQ offset)."""
    Wt = np.ascontiguousarray(np.asarray(W_out_in, np.float32).T)   # (in,out)
    K, N = Wt.shape; per = 32 // bits
    Kp = K + ((-K) % gs); Np = N + ((-N) % per)
    if Kp != K or Np != N:
        Wt = np.pad(Wt, ((0, Kp - K), (0, Np - N)))
    qw, qz, sc, _, _ = wt._gptq_quantize(Wt, gs, bits)
    return {"qweight": qw, "qzeros": qz, "scales": sc,
            "dims": np.array([K, N, Kp, Np], np.int32), "gs": gs, "bits": bits}


def dequant_int(p):
    """int packed -> fp32 (in,out). For host-numpy running (no GPU kernel)."""
    gs = int(p["gs"]); bits = int(p["bits"]); per = 32 // bits; qmax = (1 << bits) - 1
    K, N, Kp, Np = [int(v) for v in p["dims"]]
    qw = p["qweight"].astype(np.int32); qz = p["qzeros"].astype(np.int32); sc = p["scales"]
    q = np.empty((Kp, Np), np.int32)
    for r in range(per): q[r::per] = (qw >> (bits * r)) & qmax          # unpack along K
    z = np.empty((Kp // gs, Np), np.int32)
    for r in range(per): z[:, r::per] = (qz >> (bits * r)) & qmax       # unpack along N
    g = np.arange(Kp) // gs
    W = (sc[g] * (q - z[g])).astype(np.float32)
    return np.ascontiguousarray(W[:K, :N])


def quant_linear_factory(reader, bits=4, gs=128, mode="gpu", store=None):
    """Return a `linear(wname, bias)` for lm_engine.build_lm that STREAMS: reads the
    fp16 weight, quantizes it, frees the fp16. `mode='gpu'` -> QuantizedLinear (int on
    GPU, memory-efficient); `mode='numpy'` -> host dequant-matmul (for offline/no-GPU).
    If `store` (dict) given, also collects the packed ints (for saving a served model)."""
    def linear(wname, bias=True):
        W = reader(wname)                              # fp16/bf16 -> fp32 (out,in)
        p = pack_int(W, gs, bits); del W               # <-- free fp16 immediately
        bname = wname[:-7] + ".bias"
        b = reader(bname).astype(np.float32) if (bias and reader.has(bname)) else None
        if store is not None:
            store[wname[:-7]] = dict(p, **({"bias": b} if b is not None else {}))
        if mode == "gpu":
            K, N, Kp, Np = [int(v) for v in p["dims"]]
            bb = b if b is not None else np.zeros((N,), np.float32)
            ql = wt.QuantizedLinear(p["qweight"], p["qzeros"], p["scales"], bb, K, N, Kp, Np, gs, bits)
            del p
            return lambda t: ql(t)
        else:                                          # host numpy (dequant per weight, cached)
            def f(t, p=p, b=b, _c={}):
                W = _c.get("w")
                if W is None: W = dequant_int(p); _c["w"] = W
                y = t.matmul(wt.Tensor(W))
                return y + wt.Tensor(b) if b is not None else y
            return f
    return linear


# =================== offline streaming quantizer -> disk (framework-agnostic) ===================
# Produces AutoGPTQ-format safetensors (loadable by transformers+auto_gptq, vLLM, etc.),
# streaming BOTH ways: read one fp16 tensor, quantize it, write it into the current output
# shard, free it. Neither the fp16 model nor the quantized model is ever fully in RAM.
# Peak RAM = one fp16 tensor + one output shard (shard size is configurable).

def _quantize_raw(W_out_in, gs, bits):
    """(out,in) -> (q[in,out] int, scales[nG,out], zp[nG,out]).  w ~= scale*(q-zp)."""
    Wt = np.ascontiguousarray(np.asarray(W_out_in, np.float32).T)      # (in,out)=(K,N)
    K, N = Wt.shape; qmax = (1 << bits) - 1
    Kp = K + ((-K) % gs); per = 32 // bits; Np = N + ((-N) % per)
    if Kp != K or Np != N: Wt = np.pad(Wt, ((0, Kp - K), (0, Np - N)))
    nG = Kp // gs
    sc = np.empty((nG, Np), np.float32); zp = np.empty((nG, Np), np.int32); q = np.empty((Kp, Np), np.int32)
    for g in range(nG):
        blk = Wt[g * gs:(g + 1) * gs]; lo = blk.min(0); hi = blk.max(0)
        s = (hi - lo) / qmax; s[s == 0] = 1e-8
        z = np.clip(np.round(-lo / s), 0, qmax).astype(np.int32)
        sc[g] = s; zp[g] = z
        q[g * gs:(g + 1) * gs] = np.clip(np.round(blk / s) + z, 0, qmax).astype(np.int32)
    return q, sc, zp, K, N, Kp, Np

def _pack_autogptq(q, sc, zp, gs, bits):
    """Pack to AutoGPTQ layout: qweight(K/8,N) packs 8 rows/int32; qzeros(nG,N/8) packs
    8 cols/int32 storing (zp-1); scales fp16; g_idx=(i//gs). Loadable by auto_gptq/vLLM."""
    per = 32 // bits; Kp, Np = q.shape; nG = Kp // gs
    qweight = np.zeros((Kp // per, Np), np.int32)
    for r in range(per): qweight |= (q[r::per] & ((1 << bits) - 1)) << (bits * r)
    zpm = (zp - 1) & ((1 << bits) - 1)
    qzeros = np.zeros((nG, Np // per), np.int32)
    for r in range(per): qzeros |= (zpm[:, r::per]) << (bits * r)
    g_idx = (np.arange(Kp, dtype=np.int32) // gs)
    return qweight, qzeros, sc.astype(np.float16), g_idx

async def stream_quantize(read_tensor, has_tensor, tensor_names, write_shard,
                          bits=4, group_size=128, shard_bytes=512 << 20, quantize_pred=None):
    """IO-FREE streaming quantizer. Touches no filesystem — the caller injects IO via
    ASYNC callbacks (awaited, so they never block the worker):
        async read_tensor(name) -> np.ndarray (out,in)   # streaming read, one tensor at a time
        async has_tensor(name)  -> bool
        async write_shard(filename, {name: np.ndarray})   # streaming write, one shard at a time
    Neither the fp16 model nor the int model is ever fully in RAM: peak = one input tensor
    + one output shard. `quantize_pred(name)->bool` picks weights to quantize (default:
    *_proj.weight & lm_head.weight; norms/embed/router kept fp16).
    Returns a small MANIFEST {weight_map, shards, quantization_config, nq} for the caller
    to persist (index/config are small — passed as objects, not callbacks)."""
    if quantize_pred is None:
        quantize_pred = lambda n: n.endswith(".weight") and (n.endswith("_proj.weight") or n == "lm_head.weight")
    shard, sz, i, wmap, shards = {}, 0, 1, {}, []
    async def flush():
        nonlocal shard, sz, i
        if not shard: return
        fn = "model-%05d.safetensors" % i
        await write_shard(fn, shard)              # <-- caller owns the write (disk/S3/stream)
        for k in shard: wmap[k] = fn
        shards.append(fn); shard, sz, i = {}, 0, i + 1
    async def add(name, arr):
        nonlocal sz
        shard[name] = arr; sz += arr.nbytes
        if sz >= shard_bytes: await flush()
    nq = 0
    for name in tensor_names:
        if quantize_pred(name):
            W = await read_tensor(name); q, sc, zp, K, N, Kp, Np = _quantize_raw(W, group_size, bits); del W
            qw, qz, scq, gidx = _pack_autogptq(q, sc, zp, group_size, bits); del q, sc, zp
            base = name[:-7]
            await add(base + ".qweight", qw); await add(base + ".qzeros", qz)
            await add(base + ".scales", scq); await add(base + ".g_idx", gidx)
            bn = base + ".bias"
            if await has_tensor(bn): await add(bn, (await read_tensor(bn)).astype(np.float16))
            nq += 1
        else:
            await add(name, (await read_tensor(name)).astype(np.float16))
    await flush()
    return {"weight_map": wmap, "shards": shards, "nq": nq,
            "quantization_config": {"bits": bits, "group_size": group_size,
                                    "desc_act": False, "sym": False, "quant_method": "gptq"}}


async def quantize(src, dst, config, bits=4, group_size=128, shard_bytes=512 << 20,
                   names=None, quantize_pred=None):
    """Framework-compatible ASYNC convenience over the IO-free `stream_quantize`. Each of
    `src`/`dst`/`config` accepts multiple forms, auto-distinguished (via webio):
      src   : async read_tensor callback | local path (str) | safetensors bytes | {name:ndarray}
      dst   : async write_shard callback | local dir path (str)
      config: dict (or path/bytes) — small, passed as an object
    The library core does NO IO; only webio's path-adapters do (for the path forms), and
    all injected IO callbacks are awaited so nothing blocks. Returns the manifest; when
    dst is a path, also writes index.json + config.json."""
    import json as _json
    from . import webio
    read, has = webio.resolve_tensor_reader(src)
    names = names or await webio.tensor_names(src)
    write_shard = webio.resolve_shard_writer(dst)
    if isinstance(config, dict): cfg = config
    elif isinstance(config, (bytes, bytearray)): cfg = _json.loads(bytes(config))
    else: cfg = await webio.read_json(config)             # str path -> global IO callback
    manifest = await stream_quantize(read, has, names, write_shard, bits, group_size, shard_bytes, quantize_pred)
    json_dst = dst if isinstance(dst, str) else None
    await webio.write_json(json_dst, "model.safetensors.index.json",
                           {"metadata": {"total_size": 0}, "weight_map": manifest["weight_map"]})
    out_cfg = dict(cfg); out_cfg["quantization_config"] = manifest["quantization_config"]
    await webio.write_json(json_dst, "config.json", out_cfg)
    return manifest


class Quantizer:
    """Standalone quantization interface. Core (`stream`) is IO-free (inject callbacks);
    `quantize` is the framework-compatible convenience that also accepts paths/buffers."""
    stream = staticmethod(stream_quantize)     # IO-free core: (read_tensor, has, names, write_shard)
    quantize = staticmethod(quantize)          # convenience: src/dst = path | callback | buffer
    pack = staticmethod(pack_int)
    dequant = staticmethod(dequant_int)
    @staticmethod
    def pack(W_out_in, bits=4, group_size=128):
        return pack_int(W_out_in, group_size, bits)

    @staticmethod
    def quantize_hf(reader, names, bits=4, group_size=128):
        """Stream a list of weight names through the quantizer -> {base: packed}.
        `reader(name)->ndarray`, `reader.has(name)->bool`. Peak = one fp16 tensor."""
        store = {}
        for wn in names:
            if not wn.endswith(".weight"):
                continue
            W = reader(wn); store[wn[:-7]] = pack_int(W, group_size, bits); del W
            bn = wn[:-7] + ".bias"
            if reader.has(bn):
                store[wn[:-7]]["bias"] = reader(bn).astype(np.float32)
        return store
