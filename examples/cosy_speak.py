"""Text-to-speech via the GENERIC task pipeline (no model-specific interface).

    tts = await webtorch.pipeline("text-to-speech", "cosyvoice2")
    wav = tts("some text")

The same `tts(text)` call works for any TTS model; the concrete model (CosyVoice2 here:
int4 LLM + capture-replay -> flow -> vocoder) is an internal detail. Plays via Web Audio.
"""
import json, time
import numpy as np
from js import pythonIO, Object
from pyodide.ffi import to_js
import webtorch

TEXT = "Hello, this is a test of the browser port."


async def main():
    print("loading text-to-speech pipeline ...")
    tts = await webtorch.pipeline("text-to-speech", "cosyvoice2")   # generic, model-agnostic
    t0 = time.perf_counter()
    wav = tts(TEXT).astype(np.float32)                             # uniform interface
    dt = time.perf_counter() - t0
    res = {"text": TEXT, "samples": int(len(wav)), "seconds_audio": round(len(wav) / tts.sampling_rate, 2),
           "synth_s": round(dt, 1), "supports_cloning": tts.supports_cloning(), "wav_std": round(float(wav.std()), 4)}
    print("COSY_RESULT " + json.dumps(res))
    pythonIO.result = json.dumps(res)
    pythonIO.audio = to_js({"samples": wav.tolist(), "sr": tts.sampling_rate}, dict_converter=Object.fromEntries)


await main()
