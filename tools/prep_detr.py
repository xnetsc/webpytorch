"""Prepare the served DETR bundle: fold frozen BatchNorm into the ResNet-50 convs
and save a compact fp16 npz for `detection.DetrDetector.from_npz`.

Run offline (needs torch + transformers). facebook/detr-resnet-50 ships an old
config with `dilation: null`, which transformers>=5 rejects; patch the cached
config.json (dilation:false, use_pretrained_backbone:false) if load fails.

    python tools/prep_detr.py            # -> models/detr_web.npz
"""
import numpy as np, torch
from transformers import DetrForObjectDetection

OUT = "models/detr_web.npz"
m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").eval()
Z = {k: v.detach().numpy() for k, v in m.state_dict().items()}
out = {}


def fold(prefix):                                   # frozen BN -> (weight, bias)
    W = Z[prefix + ".convolution.weight"]
    g = Z[prefix + ".normalization.weight"]; b = Z[prefix + ".normalization.bias"]
    mean = Z[prefix + ".normalization.running_mean"]; var = Z[prefix + ".normalization.running_var"]
    s = g / np.sqrt(var + 1e-5)
    return (W * s[:, None, None, None]).astype(np.float32), (b - mean * s).astype(np.float32)


Wf, bf = fold("model.backbone.model.embedder.embedder"); out["emb.w"] = Wf.astype(np.float16); out["emb.b"] = bf
for si, nl in enumerate([3, 4, 6, 3]):
    for li in range(nl):
        base = f"model.backbone.model.encoder.stages.{si}.layers.{li}"
        for ci in range(3):
            Wf, bf = fold(f"{base}.layer.{ci}")
            out[f"s{si}.l{li}.c{ci}.w"] = Wf.astype(np.float16); out[f"s{si}.l{li}.c{ci}.b"] = bf
        if (base + ".shortcut.convolution.weight") in Z:
            Wf, bf = fold(f"{base}.shortcut")
            out[f"s{si}.l{li}.sc.w"] = Wf.astype(np.float16); out[f"s{si}.l{li}.sc.b"] = bf


def add(name, key, f16=True):
    a = Z[key]; out[name] = a.astype(np.float16) if f16 and a.ndim >= 2 else a.astype(np.float32)


add("inproj.w", "model.input_projection.weight"); add("inproj.b", "model.input_projection.bias", False)
add("qpos", "model.query_position_embeddings.weight")
for enc, n in [("encoder", 6), ("decoder", 6)]:
    for i in range(n):
        p = f"model.{enc}.layers.{i}"; q = f"{enc}.{i}"
        attns = ["self_attn"] + (["encoder_attn"] if enc == "decoder" else [])
        for a in attns:
            for sfx in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                add(f"{q}.{a}.{sfx}.w", f"{p}.{a}.{sfx}.weight"); add(f"{q}.{a}.{sfx}.b", f"{p}.{a}.{sfx}.bias", False)
        for a in ["self_attn_layer_norm", "final_layer_norm"] + (["encoder_attn_layer_norm"] if enc == "decoder" else []):
            add(f"{q}.{a}.w", f"{p}.{a}.weight", False); add(f"{q}.{a}.b", f"{p}.{a}.bias", False)
        for a in ["mlp.fc1", "mlp.fc2"]:
            add(f"{q}.{a}.w", f"{p}.{a}.weight"); add(f"{q}.{a}.b", f"{p}.{a}.bias", False)
add("dec.ln.w", "model.decoder.layernorm.weight", False); add("dec.ln.b", "model.decoder.layernorm.bias", False)
add("cls.w", "class_labels_classifier.weight"); add("cls.b", "class_labels_classifier.bias", False)
for i in range(3):
    add(f"bbox.{i}.w", f"bbox_predictor.layers.{i}.weight"); add(f"bbox.{i}.b", f"bbox_predictor.layers.{i}.bias", False)

np.savez(OUT, **out)
import os
print("saved", OUT, round(os.path.getsize(OUT) / 1e6, 1), "MB")
