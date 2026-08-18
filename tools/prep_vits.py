"""Prepare the served VITS bundle: fold weight_norm into the WaveNet/HiFiGAN convs
and save an fp32 npz for `tts.VitsTTS.from_npz`.

Keep this bundle fp32 -- the HiFiGAN vocoder is precision-sensitive (fp16 gives a
visibly wrong waveform). Skips the posterior_encoder (training/voice-conversion
only). Run offline (needs torch + transformers).

    python tools/prep_vits.py            # -> models/vits_web.npz
"""
import numpy as np, torch
from transformers import VitsModel

OUT = "models/vits_web.npz"
m = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
Z = {k: v.detach().numpy() for k, v in m.state_dict().items()}
out = {}
done = set()

for k in list(Z):                                   # fold weight_norm (g * v / ||v|| over dims 1,2)
    if k.endswith(".parametrizations.weight.original0"):
        pfx = k[:-len(".parametrizations.weight.original0")]
        g = Z[pfx + ".parametrizations.weight.original0"]; v = Z[pfx + ".parametrizations.weight.original1"]
        out[pfx + ".weight"] = (g * v / np.sqrt((v ** 2).sum((1, 2), keepdims=True))).astype(np.float32)
        done.add(k); done.add(pfx + ".parametrizations.weight.original1")

for k in list(Z):
    if k in done or k.startswith("posterior_encoder"):
        continue
    out[k] = Z[k].astype(np.float32)

np.savez(OUT, **out)
import os
print("saved", OUT, round(os.path.getsize(OUT) / 1e6, 1), "MB")
