"""Real quantized LLM inference with graph-capture acceleration, via the
webtorch.llm high-level API. Point MODEL at a served AutoGPTQ-Int4 dir.

Measured (Apple M-series, WebGPU): Qwen2.5-7B-Int4 -> load ~10s, TTFT ~0.6s,
decode ~21 tok/s; 3B -> ~28 tok/s. WebGL falls back to a correct slower path.
"""
import json
from js import pythonIO
from webtorch import llm, use_default_io
use_default_io()                   # REQUIRED: install IO (built-in browser fetch / host open)

MODEL = "/models/qwen7b-gptq"     # or "/models/qwen3b-gptq"
PROMPT = "Give me three tips for staying focused while working. Answer briefly."


async def main():
    model = await llm.CausalLM.from_gptq(MODEL)
    out = model.generate(PROMPT, max_new=48)
    r = {"model": MODEL, "load_s": model.load_s, "captured": model.capture_ready,
         "ttft_s": out.ttft_s, "decode_tok_s": out.decode_tok_s, "text": out.text}
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)


await main()
