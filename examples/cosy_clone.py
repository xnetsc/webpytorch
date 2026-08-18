"""Zero-shot voice cloning via the GENERIC pipeline (no model-specific interface).

    tts = await webtorch.pipeline("text-to-speech", "cosyvoice2", clone=True)
    wav = tts(text, reference_audio=(wav16, wav24), reference_text="...")

`reference_audio` is the generic voice-cloning knob — same interface for any TTS model
that supports it (`tts.supports_cloning()`). Internally: prompt audio -> generic ONNX
frontend -> speaker tokens/embedding -> int4 LLM -> flow -> vocoder. Plays via Web Audio.
"""
import io, json, time
import numpy as np
from js import pythonIO, Object
from pyodide.ffi import to_js
import webtorch
webtorch.use_default_io()          # REQUIRED: install IO (built-in browser fetch / host open)

TEXT = "Hello, this is a cloned voice speaking in the browser."


async def main():
    print("loading text-to-speech pipeline (with cloning) ...")
    tts = await webtorch.pipeline("text-to-speech", "cosyvoice2", clone=True)
    P = await webtorch.webio.load_npz("/models/cosy_prompt_wav.npz")   # generic IO helper
    t0 = time.perf_counter()
    wav = tts(TEXT, reference_audio=(P["wav16"], P["wav24"]), reference_text=str(P["prompt_text"])).astype(np.float32)
    dt = time.perf_counter() - t0
    res = {"text": TEXT, "supports_cloning": tts.supports_cloning(), "samples": int(len(wav)),
           "seconds_audio": round(len(wav) / tts.sampling_rate, 2), "synth_s": round(dt, 1),
           "wav_std": round(float(wav.std()), 4)}
    print("CLONE_RESULT " + json.dumps(res))
    pythonIO.result = json.dumps(res)
    pythonIO.audio = to_js({"samples": wav.tolist(), "sr": tts.sampling_rate}, dict_converter=Object.fromEntries)


await main()
