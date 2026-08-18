"""Real quantized LLM inference from a llama.cpp GGUF file, via webtorch.llm.
Same capture-accelerated engine as the AutoGPTQ path -- GGUF weights are
dequantized and requantized to int4 on load. Point GGUF at a served .gguf.
"""
import json
from js import pythonIO
from webtorch import llm, use_default_io
use_default_io()                   # REQUIRED: install IO (built-in browser fetch / host open)

GGUF = "/models/qwen3b-gguf/model.gguf"     # Qwen2.5-3B-Instruct q4_k_m
BITS = 4                                     # 4 or 8 -- same kernels + capture optimizations
PROMPT = "Give me three tips for staying focused while working. Answer briefly."


async def main():
    model = await llm.CausalLM.from_gguf(GGUF, bits=BITS)
    out = model.generate(PROMPT, max_new=48)
    r = {"gguf": GGUF, "bits": BITS, "load_s": model.load_s, "captured": model.capture_ready,
         "ttft_s": out.ttft_s, "decode_tok_s": out.decode_tok_s, "text": out.text}
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)


await main()
