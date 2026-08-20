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
    """Config-driven loader for the CausalLM series (+ MoE). Detects the served format and
    runs at the requested precision — **int4, int8, or fp16** — returning a ready-to-
    `.generate()` model. No model-specific code.

    `dtype`:
      - "auto"  (default): AutoGPTQ dir → int at its declared bits; GGUF → int`bits`;
        plain fp16/bf16 HF dir → run **fp16** (unquantized).
      - "fp16": force unquantized fp16 execution (plain HF dir).
      - "int4"/"int8": for a plain fp16 HF dir, quantize every linear on load; for GGUF,
        requantize to that width. (An AutoGPTQ dir always loads at its own stored bits.)
    """
    @staticmethod
    async def from_pretrained(path, dtype="auto", bits=4, lmax=320):
        from . import llm as _llm, webio
        p = path.rstrip("/")
        if p.endswith(".gguf"):
            gb = 8 if dtype == "int8" else (4 if dtype == "int4" else bits)
            # dtype="fp16" runs a GGUF unquantized (works without a GPU; the int kernel is
            # GPU-only). "auto"/int4/int8 requantize to the int engine as before.
            return await _llm.CausalLM.from_gguf(p, lmax=lmax, bits=gb,
                                                 quantize=(dtype != "fp16"))
        cfg = await webio.read_json(p + "/config.json")
        if "quantization_config" in cfg:                     # already-quantized AutoGPTQ (int4/int8)
            return await _llm.CausalLM.from_gptq(p, lmax=lmax)
        q = {"int4": 4, "int8": 8}.get(dtype)                # plain fp16/bf16 HF dir
        return await _llm.CausalLM.from_fp16(p, lmax=lmax, quantize=q)   # q=None → run fp16


# ---- ONE unified entry point for every model type ---------------------------
class Model:
    """Uniform handle returned by `webtorch.load(...)`, whatever the model is.

    The specialised APIs (`AutoModelForCausalLM`, `pipeline`, `OnnxModel`, …) stay available —
    this is the single door that covers all of them, so callers do not have to know which one a
    given model needs:

        m = await webtorch.load("Qwen/Qwen3-0.6B")            # LLM (repo id / dir / .gguf)
        print(m.generate("Hello", max_new=64))
        print(m("Hello"))                                     # calling it does the natural thing

        m = await webtorch.load("/models/any.onnx")            # ONNX graph
        outs = m.run({"input": arr})

        m = await webtorch.load(task="text-to-speech")         # any registered pipeline/task
        wav = m("hello")

        m = await webtorch.load("/models/vlm", encoder="my-vision")   # multimodal
        print(m.generate("Describe this.", media=img))

    `.kind` says what was loaded ("causal-lm" / "onnx" / a task name), `.impl` is the underlying
    object, and any attribute not defined here is forwarded to it — so nothing is hidden."""

    def __init__(self, impl, kind):
        self.impl = impl; self.kind = kind

    def __getattr__(self, name):                 # forward everything else to the impl
        return getattr(self.__dict__["impl"], name)

    def __repr__(self):
        return "<webtorch.Model kind=%s impl=%s>" % (self.kind, type(self.impl).__name__)

    # inference verbs, in the order they are tried by the unified `infer`
    _VERBS = ("generate", "run", "synth", "detect", "transcribe", "classify", "encode")

    def infer(self, *a, **kw):
        """**The unified inference call** — the same method for every model type.

            m.infer("Hello", max_new=32)          # LLM      -> text
            m.infer(text)                          # TTS      -> waveform
            m.infer(audio)                         # ASR      -> text
            m.infer(image, threshold=0.3)          # detection-> boxes
            m.infer({"input": arr})                # ONNX     -> outputs
            m.infer("Describe this.", media=img)   # multimodal

        It dispatches to whatever the underlying model exposes (`__call__`, then `generate` /
        `run` / `synth` / `detect` / `transcribe` / `classify` / `encode`), so callers never
        branch on model type. `m(...)` is a shorthand for this."""
        impl = self.impl
        if callable(impl):
            return impl(*a, **kw)
        for m in self._VERBS:
            f = getattr(impl, m, None)
            if f is not None:
                return f(*a, **kw)
        raise TypeError("don't know how to run %r — it exposes none of %s"
                        % (type(self.impl), ", ".join(self._VERBS)))

    # `run` is an alias so ONNX-style callers keep working through the same door
    def run(self, *a, **kw):
        return self.infer(*a, **kw)

    def stream(self, *a, **kw):
        """Streaming inference where the model supports it (LLMs); otherwise yields the single
        non-streaming result, so callers can always use the same loop."""
        f = getattr(self.impl, "stream", None)
        if f is not None:
            return f(*a, **kw)
        return iter([self.infer(*a, **kw)])

    def __call__(self, *a, **kw):
        return self.infer(*a, **kw)


async def load(source=None, task=None, dtype="auto", encoder=None, **kw):
    """**The unified loader.** Point it at anything and get a `Model` back:

      source: a model dir / repo id ("org/repo" with a hub reader installed), a `.gguf` file,
              a `.onnx` file, or omitted when `task=` names a registered pipeline.
      task:   force a task ("text-to-speech", "asr", "object-detection", …); with `source` it
              selects the pipeline's model name.
      dtype:  "auto" | "fp16" | "int4" | "int8" (LLMs).
      encoder: name of a registered media encoder -> a multimodal model.

    Detection is by content, not by model name: `.onnx`/`.gguf` by extension, otherwise the
    served `config.json` decides (a decoder config -> the generic CausalLM/MoE/hybrid engine).
    Every specialised API remains available and unchanged."""
    from . import multimodal, webio
    if source is None:
        if task is None:
            raise ValueError("load() needs a `source` (path/repo/file) or a `task=`")
        return Model(await pipeline(task, kw.pop("model", "auto"), **kw), task)

    src = str(source).rstrip("/")
    if src.endswith(".onnx"):
        return Model(await OnnxModel.from_source(src), "onnx")
    if task is not None:                                   # explicit task -> registry
        return Model(await pipeline(task, kw.pop("model", "auto"), path=src, **kw), task)
    lm = await AutoModelForCausalLM.from_pretrained(
        src, dtype=dtype, **{k: kw[k] for k in ("bits", "lmax") if k in kw})
    if encoder is not None:                                # decoder + media encoder
        enc = await multimodal.load_encoder(encoder, **kw.get("encoder_kwargs", {}))
        return Model(multimodal.MultimodalLM(lm, enc, placeholder_id=kw.get("placeholder_id")),
                     "multimodal")
    return Model(lm, "causal-lm")


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

class _SpeechToText:
    """`pipe(audio)` -> text. Audio is a waveform (ndarray/list) or whatever the impl accepts;
    `sampling_rate=` is forwarded when given. Protocol: `.transcribe(audio, …) -> str`."""
    def __init__(self, impl): self._impl = impl
    @property
    def sampling_rate(self): return getattr(self._impl, "sr", 16000)
    def __call__(self, audio, sampling_rate=None, **kw):
        if sampling_rate is not None: kw["sampling_rate"] = sampling_rate
        return self._impl.transcribe(audio, **kw)

class _AudioClassification:
    """`pipe(audio)` -> labels/scores. Protocol: `.classify(audio, …)`."""
    def __init__(self, impl): self._impl = impl
    @property
    def sampling_rate(self): return getattr(self._impl, "sr", 16000)
    def __call__(self, audio, **kw): return self._impl.classify(audio, **kw)

class _Generic:
    """Fallback wrapper for a task webtorch has no built-in protocol for. It forwards the call
    to the impl (`impl(...)`, else `impl.run(...)`) and proxies attribute access, so a third
    party can register an entirely NEW task type without changing the SDK."""
    def __init__(self, impl): self._impl = impl
    def __call__(self, *a, **kw):
        f = self._impl if callable(self._impl) else getattr(self._impl, "run", None)
        if f is None:
            raise TypeError("pipeline impl %r is not callable and has no .run()" % type(self._impl))
        return f(*a, **kw)
    def __getattr__(self, name): return getattr(self._impl, name)

_WRAPPERS = {"text-to-speech": _TextToSpeech, "object-detection": _ObjectDetection,
             "image-to-text": _ImageToText, "text-generation": _TextGeneration,
             "automatic-speech-recognition": _SpeechToText,
             "audio-classification": _AudioClassification}
_TASK_ALIASES = {"tts": "text-to-speech", "detection": "object-detection",
                 "visual-question-answering": "image-to-text", "vl": "image-to-text",
                 "causal-lm": "text-generation",
                 "asr": "automatic-speech-recognition",
                 "speech-to-text": "automatic-speech-recognition",
                 "stt": "automatic-speech-recognition"}
_REGISTRY = {t: {} for t in _WRAPPERS}
_DEFAULTS = {}

def register_task(task, wrapper):
    """Register a NEW task type (its uniform call interface). `wrapper(impl)` returns the
    object `pipeline()` hands back. Only needed for a task webtorch has no protocol for —
    otherwise a generic forwarding wrapper is used automatically."""
    _WRAPPERS[_TASK_ALIASES.get(task, task)] = wrapper

def register_pipeline(task, name, loader, default=False):
    """Register a model for a task so `pipeline(task, name)` can load it. `loader(**kw)`
    is async and returns an impl implementing the task protocol. Third-party models — and
    entirely new task names — plug in here without modifying the SDK."""
    task = _TASK_ALIASES.get(task, task)
    _REGISTRY.setdefault(task, {})[name] = loader
    if default or task not in _DEFAULTS: _DEFAULTS[task] = name

def list_pipelines(task=None):
    """Registered models: `{task: [names]}`, or `[names]` for one task. Lets callers discover
    what is available instead of hard-coding model names."""
    if task is not None:
        return sorted(_REGISTRY.get(_TASK_ALIASES.get(task, task), {}))
    return {t: sorted(m) for t, m in _REGISTRY.items() if m}

async def pipeline(task, model="auto", **kw):
    """transformers-style task pipeline -> a uniform task object (same methods across
    models). The concrete model is internal; add your own via `register_pipeline` (and a new
    task type via `register_task`). Unknown task names still work through a generic wrapper."""
    task = _TASK_ALIASES.get(task, task)
    reg = _REGISTRY.get(task)
    if not reg:
        raise ValueError("no model registered for task %r (registered: %s) — add one with "
                         "register_pipeline(task, name, loader)" % (task, sorted(list_pipelines())))
    name = _DEFAULTS.get(task) if model in ("auto", None) else model
    if name not in reg:
        raise ValueError("no model %r for task %r (registered: %s)" % (name, task, sorted(reg)))
    impl = await reg[name](**kw)
    return _WRAPPERS.get(task, _Generic)(impl)


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

async def _load_multimodal(**kw):
    """Generic multimodal impl: ANY decoder + ANY registered media encoder.
        pipeline("image-to-text", "auto", path=…, encoder="my-vision")
    The decoder is loaded by config (the whole CausalLM/MoE family) and paired with the named
    encoder through `MultimodalLM` — no model-specific code."""
    from . import multimodal
    lm = await AutoModelForCausalLM.from_pretrained(
        kw["path"], **{k: kw[k] for k in ("dtype", "bits", "lmax") if k in kw})
    enc = await multimodal.load_encoder(kw.get("encoder", "auto"),
                                        **kw.get("encoder_kwargs", {}))
    return multimodal.MultimodalLM(lm, enc, placeholder_id=kw.get("placeholder_id"))

register_pipeline("text-to-speech", "cosyvoice2", _load_cosyvoice2, default=True)
register_pipeline("text-to-speech", "vits", _load_vits)
register_pipeline("text-to-speech", "mms-tts", _load_vits)
register_pipeline("object-detection", "yolo", _load_yolo, default=True)
register_pipeline("object-detection", "detr", _load_detr)
register_pipeline("image-to-text", "qwen-vl", _load_qwenvl, default=True)
register_pipeline("image-to-text", "auto", _load_multimodal)     # any decoder + any encoder
register_pipeline("text-generation", "auto", _load_causal, default=True)


# ---- generic ONNX (any model) -----------------------------------------------
class OnnxModel:
    @staticmethod
    async def from_source(src, io=None):
        from . import onnxrt
        return await onnxrt.OnnxModel.from_source(src, io)

    @staticmethod
    async def from_url(url):                     # back-compat alias
        return await OnnxModel.from_source(url)


# ---- symmetric global async IO callbacks (REQUIRED; single injection point for ALL reads AND writes) ----
from .webio import (set_io_read, get_io_read, io_read, set_io_write, get_io_write, io_write,
                    use_default_io, default_io_read, default_io_write, hf_read, modelscope_read,
                    make_cached_reader, http_get, http_size, HttpError,
                    default_cache_dir, list_cache, cache_hosts, cache_size, read_cache,
                    write_cache, delete_cache, clear_cache)


# ---- explicit exports --------------------------------------------------------
__all__ = ["install_torch", "load", "Model", "AutoTokenizer", "AutoModelForCausalLM", "Quantizer", "pipeline", "register_pipeline",
           "register_task", "list_pipelines", "OnnxModel", "set_io_read", "get_io_read", "io_read", "set_io_write", "get_io_write", "io_write",
           "use_default_io", "default_io_read", "default_io_write", "hf_read", "modelscope_read",
           "make_cached_reader", "http_get", "http_size", "HttpError",
           "default_cache_dir", "list_cache", "cache_hosts", "cache_size", "read_cache",
           "write_cache", "delete_cache", "clear_cache"]
