"""webtorch SDK — a framework-compatible facade over the whole library.

Goal: third parties use webtorch through the SAME interfaces they already know
from PyTorch and HuggingFace transformers, with generic / series-level entry
points (never one model-specific API):

    import webtorch.sdk as W
    W.install_torch()                       # `import torch` now resolves to webtorch
    tok  = W.AutoTokenizer.from_pretrained("/models/qwen3b-gptq")
    lm   = await W.AutoModelForCausalLM.from_pretrained("/models/qwen3b-gptq")  # dense OR MoE
    out  = lm.generate("Hello", max_new=64)

    tts  = await W.pipeline("text-to-speech", "cosyvoice2")   # CosyVoice2 / VITS
    wav  = tts("some text")
    det  = await W.pipeline("object-detection", "yolo")       # YOLO / DETR
    onnx = await W.OnnxModel.from_url("/models/any.onnx")     # run ANY onnx

Everything dispatches by config/task — the CausalLM *series* (Qwen2/Qwen3/Llama),
the MoE *series* (Qwen2-MoE/Qwen3-MoE via one generic MoE layer), the ONNX runner
(any graph), and the audio/vision pipelines are all generic building blocks.
"""
import json
import numpy as np


# ---- torch compatibility: `import torch` -> webtorch -------------------------
def install_torch():
    """Register webtorch's torch-compatible shim so `import torch` works (nn,
    autograd, optim, Tensor ops). Mirrors torch 2.x API. Idempotent."""
    from . import torchshim
    torchshim.install()
    return __import__("torch")


# ---- transformers-style tokenizer -------------------------------------------
class AutoTokenizer:
    """Generic byte-BPE tokenizer loader (Qwen/GPT-2 family). Accepts a served
    dir with vocab.json+merges.txt, or a {vocab,merges} json url."""
    @staticmethod
    async def from_pretrained(path):
        from . import llm as _llm, webio
        base = path.rstrip("/")
        try:
            v = await webio.read_json(base + "/vocab.json")
            mtxt = await webio.read_text(base + "/merges.txt")
            merges = [m for m in mtxt.split("\n")[1:] if m and not m.startswith("#")]
        except Exception:
            tj = await webio.read_json(base if base.endswith(".json") else base + ".json")
            v, merges = tj["vocab"], tj["merges"]
        return _llm.BPETokenizer(v, merges)


# ---- transformers-style causal LM (dense + MoE series) ----------------------
class AutoModelForCausalLM:
    """Config-driven loader for the CausalLM series. Detects the served format
    (AutoGPTQ int4 safetensors / llama.cpp GGUF) and MoE (config num_experts),
    returning a ready-to-`.generate()` model. No model-specific code."""
    @staticmethod
    async def from_pretrained(path, bits=4, lmax=320):
        from . import llm as _llm
        p = path.rstrip("/")
        if p.endswith(".gguf"):
            return await _llm.CausalLM.from_gguf(p, lmax=lmax, bits=bits)
        return await _llm.CausalLM.from_gptq(p, lmax=lmax)   # handles dense; MoE via build_moe_lm


# Dedicated quantization interface (IO-free core `Quantizer.stream(...)` + framework-
# compatible `Quantizer.quantize(src, dst, config, ...)` accepting path|callback|bytes).
# Output is AutoGPTQ int4/int8, loadable by auto_gptq / vLLM / transformers / this SDK.
from . import quantize as _q
Quantizer = _q.Quantizer


# ---- generic task pipelines: protocol + registry (extensible, no per-model branch) ----
# A pipeline wrapper only ever calls its task's PROTOCOL on the model impl; it never
# branches on which model it is. New models plug in by registering a loader that returns
# an impl implementing the protocol — no SDK code changes needed.
#
#   TTS protocol            : .sr ; .synth(text)->waveform ; .clone(text, ref, ref_text)->waveform ; .can_clone
#   object-detection        : .detect(image, threshold=…)
#   image-to-text           : .generate(prompt, image=…)
#   text-generation         : .generate(prompt, …) ; .stream(prompt, …)

class _TextToSpeech:
    """`pipe(text)` -> waveform; `pipe(text, reference_audio=…, reference_text=…)` clones
    (generic knob; models without `.can_clone` raise NotImplementedError)."""
    def __init__(self, impl): self._impl = impl
    @property
    def sampling_rate(self): return getattr(self._impl, "sr", 16000)
    def supports_cloning(self): return bool(getattr(self._impl, "can_clone", False))
    def __call__(self, text, reference_audio=None, reference_text="", **kw):
        if reference_audio is not None:
            if not self.supports_cloning():
                raise NotImplementedError("this TTS model does not support voice cloning")
            return self._impl.clone(text, reference_audio, reference_text, **kw)
        return self._impl.synth(text, **kw)

class _ObjectDetection:
    def __init__(self, impl): self._impl = impl
    def __call__(self, image, threshold=None, **kw):
        return self._impl.detect(image, **({} if threshold is None else {"threshold": threshold}), **kw)

class _ImageToText:
    def __init__(self, impl): self._impl = impl
    def __call__(self, image, prompt="Describe the image.", **kw):
        return self._impl.generate(prompt, image=image, **kw)

class _TextGeneration:
    def __init__(self, impl): self._impl = impl
    def __call__(self, prompt, **kw): return self._impl.generate(prompt, **kw)
    def stream(self, prompt, **kw): return self._impl.stream(prompt, **kw)

_WRAPPERS = {"text-to-speech": _TextToSpeech, "object-detection": _ObjectDetection,
             "image-to-text": _ImageToText, "text-generation": _TextGeneration}
_TASK_ALIASES = {"tts": "text-to-speech", "detection": "object-detection",
                 "visual-question-answering": "image-to-text", "vl": "image-to-text",
                 "causal-lm": "text-generation"}
_REGISTRY = {t: {} for t in _WRAPPERS}
_DEFAULTS = {}

def register_pipeline(task, name, loader, default=False):
    """Register a model for a task so `pipeline(task, name)` can load it. `loader(**kw)`
    is async and returns an impl implementing the task protocol. Third-party models plug
    in here without modifying the SDK."""
    task = _TASK_ALIASES.get(task, task)
    _REGISTRY.setdefault(task, {})[name] = loader
    if default or task not in _DEFAULTS: _DEFAULTS[task] = name

async def pipeline(task, model="auto", **kw):
    """transformers-style task pipeline -> a uniform task object (same methods across
    models). The concrete model is internal; add your own via `register_pipeline`."""
    task = _TASK_ALIASES.get(task, task)
    reg = _REGISTRY.get(task)
    if reg is None: raise ValueError("unknown task: %s (have %s)" % (task, list(_REGISTRY)))
    name = _DEFAULTS.get(task) if model in ("auto", None) else model
    if name not in reg:
        raise ValueError("no model %r for task %r (registered: %s)" % (name, task, list(reg)))
    impl = await reg[name](**kw)
    return _WRAPPERS[task](impl)


# ---- built-in model loaders (registered; each returns a protocol-conforming impl) ----
async def _load_cosyvoice2(**kw):
    from . import cosyvoice
    m = await cosyvoice.CosyVoice2TTS.from_npz(kw.get("flow", "/models/cosy_flow.npz"),
                                               kw.get("hift", "/models/cosy_hift.npz"),
                                               kw.get("baked", "/models/cosy_baked.npz"))
    if kw.get("llm", True):
        await m.load_llm(kw.get("llm_npz", "/models/cosy_llm.npz"), kw.get("tok", "/models/cosy_qwen_tok.json"))
    if kw.get("clone", False):
        await m.load_clone(kw.get("spk_tok", "/models/speech_tokenizer_v2.onnx"),
                           kw.get("campplus", "/models/campplus.onnx"), kw.get("melfilters", "/models/cosy_melfilters.npz"))
    return m

async def _load_vits(**kw):
    from . import tts
    return await tts.VitsTTS.from_npz(kw.get("npz", "/models/vits_web.npz"))

async def _load_detr(**kw):
    from . import detection
    return await detection.DetrDetector.from_npz(kw.get("npz", "/models/detr_web.npz"))
async def _load_yolo(**kw):
    from . import detection
    return await detection.YoloDetector.from_npz(kw.get("npz", "/models/yolo_web.npz"))
async def _load_qwenvl(**kw):
    from . import vl
    return await vl.QwenVL.from_pretrained(kw.get("path", "/models/qwen2.5-vl-3b"))
async def _load_causal(**kw):
    return await AutoModelForCausalLM.from_pretrained(kw["path"], **{k: kw[k] for k in ("bits", "lmax") if k in kw})

register_pipeline("text-to-speech", "cosyvoice2", _load_cosyvoice2, default=True)
register_pipeline("text-to-speech", "vits", _load_vits)
register_pipeline("text-to-speech", "mms-tts", _load_vits)
register_pipeline("object-detection", "yolo", _load_yolo, default=True)
register_pipeline("object-detection", "detr", _load_detr)
register_pipeline("image-to-text", "qwen-vl", _load_qwenvl, default=True)
register_pipeline("text-generation", "auto", _load_causal, default=True)


# ---- generic ONNX (any model) -----------------------------------------------
class OnnxModel:
    @staticmethod
    async def from_url(url):
        from . import onnxrt
        return await onnxrt.OnnxModel.from_url(url)


# ---- symmetric global async IO callbacks (REQUIRED; single injection point for ALL reads AND writes) ----
from .webio import (set_io_read, get_io_read, io_read, set_io_write, get_io_write, io_write,
                    use_default_io, default_io_read, default_io_write)


# ---- explicit exports --------------------------------------------------------
__all__ = ["install_torch", "AutoTokenizer", "AutoModelForCausalLM", "Quantizer", "pipeline", "register_pipeline",
           "OnnxModel", "set_io_read", "get_io_read", "io_read", "set_io_write", "get_io_write", "io_write",
           "use_default_io", "default_io_read", "default_io_write"]
