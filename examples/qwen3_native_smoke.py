"""End-to-end native-width smoke test using the repository's local Qwen3 GGUF.

The file is fetched from this checkout through the dev server.  Browser persistence is
disabled deliberately: a repository file must not be copied into the model cache merely
to test it.  ``weights="native"`` keeps every supported GGUF tensor in its stored format.
"""
import json

from js import pythonIO

from webtorch import core as wt
from webtorch import llm, use_default_io


GGUF = "/models/Qwen3-0.6B-Q4_K_M.gguf"


def native_formats(model):
    """Count packed linears without assuming a model name or architecture."""
    counts = {}
    seen = set()
    stack = [model.layers]
    while stack:
        value = stack.pop()
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(value, wt.GGMLLinear):
            name = value.storage_format
            counts[name] = counts.get(name, 0) + 1
        elif isinstance(value, wt.GGMLMoELinear):
            name = value.type_name
            counts[name] = counts.get(name, 0) + 1
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return counts


async def main():
    use_default_io(cache=False)
    model = await llm.CausalLM.from_gguf(GGUF, weights="native")
    formats = native_formats(model)
    out = model.generate(
        "Reply with exactly the single word OK.",
        max_new=8,
        do_sample=False,
        enable_thinking=False,
    )
    text = out.text.strip()
    result = {
        "ok": bool(formats) and text.upper().rstrip(".!\n ") == "OK",
        "gguf": GGUF,
        "weights": "native",
        "native_formats": formats,
        "load_s": model.load_s,
        "captured": model.capture_ready,
        "ttft_s": out.ttft_s,
        "decode_tok_s": out.decode_tok_s,
        "text": text,
    }
    print("RESULT " + json.dumps(result))
    pythonIO.result = json.dumps(result)


await main()
