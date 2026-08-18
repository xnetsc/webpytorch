"""Text-to-speech (VITS / MMS-TTS-eng) in the browser via webtorch.tts.

Synthesizes a waveform for a text prompt (HiFiGAN vocoder on webtorch GPU) and
verifies it against the HF-transformers reference waveform. Prints TTS_RESULT.
"""
import io, json, time
import numpy as np
from js import pythonIO
from pyodide.http import pyfetch
from webtorch import tts, use_default_io
use_default_io()                   # REQUIRED: install IO (built-in browser fetch / host open)

NPZ = "/models/vits_web.npz"
REF = "/models/vits_ref_wav.npy"
TEXT = "hello world"


async def main():
    v = await tts.VitsTTS.from_npz(NPZ)
    ids = v.tokenize(TEXT)
    t0 = time.perf_counter()
    wav = v.synthesize(ids)
    dt = time.perf_counter() - t0
    r = await pyfetch(REF)
    ref = np.load(io.BytesIO(bytes(await r.bytes())))
    n = min(len(wav), len(ref))
    err = float(np.abs(wav[:n] - ref[:n]).max())
    res = {"text": TEXT, "n_tokens": len(ids), "samples": int(len(wav)),
           "seconds_audio": round(len(wav) / v.sr, 2), "synth_s": round(dt, 2),
           "max_abs_err_vs_hf": err, "ok": bool(err < 5e-3 and len(wav) == len(ref)),
           "wav_min": round(float(wav.min()), 3), "wav_max": round(float(wav.max()), 3)}
    print("TTS_RESULT " + json.dumps(res))
    pythonIO.result = json.dumps(res)


await main()
