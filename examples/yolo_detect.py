"""YOLOv8n object detection in the browser via webtorch.detection.YoloDetector.

CSP backbone + PAN neck + DFL head (GPU convs) + host NMS. Detects objects in the
COCO demo image and prints YOLO_RESULT (labels/scores/boxes + timing).
"""
import io, json, time
from js import pythonIO
import pyodide_js
from pyodide.http import pyfetch
from webtorch import detection

NPZ = "/models/yolov8n_web.npz"
NAMES = "/models/coco_names.json"
IMAGE = "/models/detr_cats.jpg"


async def main():
    await pyodide_js.loadPackage("pillow")
    from PIL import Image
    r = await pyfetch(IMAGE)
    img = Image.open(io.BytesIO(bytes(await r.bytes())))
    det = await detection.YoloDetector.from_npz(NPZ, NAMES)
    # warm run (compiles kernels / fills weight cache), then timed runs
    det.detect(img, threshold=0.25, imgsz=640)
    import numpy as np
    pv, r = det.preprocess(img, 640)
    tf = time.perf_counter(); det.forward(pv); forward_s = time.perf_counter() - tf
    t0 = time.perf_counter()
    results = det.detect(img, threshold=0.25, imgsz=640)
    dt = time.perf_counter() - t0
    res = {"infer_s": round(dt, 3), "forward_s": round(forward_s, 3),
           "n": len(results), "detections": results}
    print("YOLO_RESULT " + json.dumps(res))
    pythonIO.result = json.dumps(res)


await main()
