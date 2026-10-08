"""Low-rank adapters (PEFT LoRA), read from the PEFT library's own files.

A PEFT adapter is `adapter_config.json` (rank, alpha, which modules) and
`adapter_model.safetensors` (each targeted module's A and B). Both are the library's format,
so an adapter saved by it reads the same way whatever model it was trained for; nothing here
knows a model. What the adapter changes is named by the Hugging Face module path it was
trained on ("model.layers.3.self_attn.q_proj"); the model it is attached to decides where
that module lives (`CausalLM.attach_adapter`).

What this does not read, it refuses rather than skips: an adapter applied to part of what it
was trained on gives answers that look fine and are not.
"""
import json
import math
import re

import numpy as np

ADAPTER_CONFIG = "adapter_config.json"     # PEFT's file names, not a model's
ADAPTER_WEIGHTS = "adapter_model.safetensors"

_NAME = re.compile(r"^(?:base_model\.model\.)?(?P<path>.+?)\.lora_(?P<ab>[AB])"
                   r"(?:\.[A-Za-z0-9_]+)?\.weight$")


def _pattern_value(patterns, path, default):
    """PEFT's per-module override: a key matches a module whose path ends in it, or that it
    matches as a regular expression."""
    for key, value in (patterns or {}).items():
        if path == key or path.endswith("." + key) or re.fullmatch(key, path):
            return value
    return default


def lora_scale(cfg, path):
    """The factor PEFT multiplies B @ A by for `path`: alpha / r, or alpha / sqrt(r) with
    rsLoRA, after the per-module rank and alpha overrides."""
    r = int(_pattern_value(cfg.get("rank_pattern"), path, cfg.get("r", 8)))
    alpha = float(_pattern_value(cfg.get("alpha_pattern"), path, cfg.get("lora_alpha", r)))
    if r <= 0:
        raise ValueError("LoRA rank %r for %s" % (r, path))
    return alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)


# PEFT options that do not change what an adapter computes once trained: bookkeeping, how
# the weights were initialised, which modules were targeted (the weights file says that),
# and the ones read above. Any other option set to something is refused below, because a
# newer PEFT feature that changes the forward pass arrives as exactly such an option.
_INERT = {"peft_type", "peft_version", "task_type", "revision", "base_model_name_or_path",
          "auto_mapping", "inference_mode", "r", "lora_alpha", "lora_dropout", "rank_pattern",
          "alpha_pattern", "use_rslora", "target_modules", "exclude_modules",
          "layers_to_transform", "layers_pattern", "init_lora_weights", "loftq_config",
          "eva_config", "corda_config", "lora_ga_config", "megatron_config", "megatron_core",
          "qalora_group_size", "ensure_weight_tying", "bias"}


def check_config(cfg):
    """Raise for the PEFT options this engine does not apply."""
    if str(cfg.get("peft_type", "LORA")).upper() != "LORA":
        raise NotImplementedError("adapter type %r: only LoRA is applied" % cfg.get("peft_type"))
    for key, why in (("use_dora", "DoRA rescales each column by a learned magnitude"),
                     ("fan_in_fan_out", "its weights are stored transposed"),
                     ("lora_bias", "it carries a bias of its own"),
                     ("modules_to_save", "it replaces whole modules"),
                     ("alora_invocation_tokens", "it applies only after its invocation tokens"),
                     ("trainable_token_indices", "it retrains some token embeddings"),
                     ("layer_replication", "it repeats layers")):
        if cfg.get(key):
            raise NotImplementedError("LoRA with %s set is not applied: %s" % (key, why))
    if str(cfg.get("bias", "none")) != "none":
        raise NotImplementedError("LoRA bias=%r is not applied" % cfg.get("bias"))
    unknown = sorted(k for k, v in cfg.items() if v and k not in _INERT)
    if unknown:
        raise NotImplementedError("LoRA options this engine does not apply: %s"
                                  % ", ".join(unknown))


def adapter_for(source, adapter=None, container=None):
    """The adapter folder a model load should attach, or None.

    `adapter` names it, or False says none. Left unset, a model whose folder holds PEFT's
    two files at its top level is that model with that adapter: PEFT saves an adapter as a
    folder of its own, so its files sit beside a model's only when someone put them there.
    An adapter in a subfolder may be an optional extra and is attached only when named."""
    if adapter is False:
        return None
    if adapter:
        return str(adapter).rstrip("/")
    from . import webio
    s = str(source).rstrip("/")
    folder = (s.rsplit("/", 1)[0] if "/" in s else None) if container else s
    if not folder:
        return None
    names = webio.files_under(folder) or ()
    if ADAPTER_CONFIG in names and ADAPTER_WEIGHTS in names:
        return folder
    return None


async def _read(path, offset, length):
    from . import webio
    return bytes(await webio.io_read(path, offset, length))


async def read_lora(folder):
    """{Hugging Face module path: (A, B, scale)} from the PEFT LoRA adapter in `folder`, A as
    (rank, in) and B as (out, rank) in float32."""
    from . import webio
    from .hfcompat import _decode
    folder = str(folder).rstrip("/")
    cfg = await webio.read_json(folder + "/" + ADAPTER_CONFIG)
    check_config(cfg)
    path = folder + "/" + ADAPTER_WEIGHTS
    n = int.from_bytes(await _read(path, 0, 8), "little")
    header = json.loads((await _read(path, 8, n)).decode("utf-8"))
    header.pop("__metadata__", None)
    base = 8 + n
    halves = {}
    for name, info in header.items():
        m = _NAME.match(name)
        if m is None:
            raise NotImplementedError("adapter tensor %r is not a LoRA A or B weight" % name)
        a, z = info["data_offsets"]
        arr = _decode(await _read(path, base + a, z - a), info["dtype"], info["shape"])
        halves.setdefault(m.group("path"), {})[m.group("ab")] = np.asarray(arr, np.float32)
    out = {}
    for module, ab in sorted(halves.items()):
        if set(ab) != {"A", "B"}:
            raise ValueError("adapter module %s has only lora_%s" % (module, "".join(ab)))
        A, B = ab["A"], ab["B"]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            raise ValueError("adapter module %s: A %s and B %s do not share a rank"
                             % (module, A.shape, B.shape))
        out[module] = (A, B, lora_scale(cfg, module))
    if not out:
        raise ValueError("%s holds no LoRA weights" % path)
    return out
