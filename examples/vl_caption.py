"""End-to-end Qwen2.5-VL image captioning in the browser via webtorch.vl.

Streams the VL model (LM + ViT quantized to int4 on load), preprocesses an image,
runs the vision tower + M-RoPE LM, and greedily decodes a caption. WebGPU uses the
capture-accelerated decode; WebGL uses a correct fresh-forward fallback.
"""
import io, json, time
from js import pythonIO
import pyodide_js
from pyodide.http import pyfetch
from webtorch import vl, use_default_io
use_default_io()                   # REQUIRED: install IO (built-in browser fetch / host open)

MODEL = "/models/qwen2.5-vl-3b"
IMAGE = "/models/vl_test.png"
PROMPT = "Describe this image in one sentence."


async def main():
    await pyodide_js.loadPackage("pillow")
    from PIL import Image
    r = await pyfetch(IMAGE)
    img = Image.open(io.BytesIO(bytes(await r.bytes())))
    print("image", img.size)
    t0 = time.perf_counter()
    model = await vl.VLCausalLM.from_qwen2_5_vl(MODEL, lmax=1024, bits=4)
    print("loaded in %.1fs" % (time.perf_counter() - t0))
    out = model.generate(PROMPT, image=img, max_new=48)
    res = {"load_s": model.load_s, "captured": model.capture_ready,
           "ttft_s": out.ttft_s, "decode_tok_s": out.decode_tok_s, "text": out.text}
    print("VL_RESULT " + json.dumps(res))
    pythonIO.result = json.dumps(res)


await main()
