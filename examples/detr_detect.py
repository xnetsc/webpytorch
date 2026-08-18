"""DETR object detection in the browser via webtorch.detection.

Loads a folded-BN DETR (ResNet-50 + transformer) from a served .npz, detects
objects in the COCO demo image, prints DETR_RESULT json (labels/scores/boxes).
"""
import io, json, time
from js import pythonIO
import pyodide_js
from pyodide.http import pyfetch
from webtorch import detection, use_default_io
use_default_io()                   # REQUIRED: install IO (built-in browser fetch / host open)

NPZ = "/models/detr_web.npz"
IMAGE = "/models/detr_cats.jpg"


async def main():
    await pyodide_js.loadPackage("pillow")
    from PIL import Image
    r = await pyfetch(IMAGE)
    img = Image.open(io.BytesIO(bytes(await r.bytes())))
    print("image", img.size)
    t0 = time.perf_counter()
    det = await detection.DetrDetector.from_npz(NPZ)
    print("loaded in %.1fs" % (time.perf_counter() - t0))
    t1 = time.perf_counter()
    results = det.detect(img, threshold=0.7)
    dt = time.perf_counter() - t1
    res = {"infer_s": round(dt, 2), "n": len(results), "detections": results}
    print("DETR_RESULT " + json.dumps(res))
    pythonIO.result = json.dumps(res)


await main()
