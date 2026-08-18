"""Train a small CNN with webtorch autograd on the GPU (WebGPU or WebGL).
Synthetic 2-class image task; loss should fall and accuracy rise -- shows that
Conv2d / BatchNorm2d / MaxPool2d / Linear + cross_entropy + Adam all train."""
import json
import numpy as np
from js import pythonIO
import cupy as cp
from webtorch import core as wt


def main():
    r = {"backend": cp.get_backend_name()}
    np.random.seed(0)
    N, C, H, W = 64, 1, 12, 12
    X = (np.random.rand(N, C, H, W) * 0.2).astype(np.float32)
    y = np.random.randint(0, 2, N)
    X[y == 1, :, 3:8, 3:8] += 0.8                       # class 1 has a bright blob
    Xt = wt.Tensor(X)

    model = wt.Sequential(
        wt.Conv2d(1, 8, 3, padding=1), wt.BatchNorm2d(8), wt.ReLU(), wt.MaxPool2d(2),
        wt.Conv2d(8, 16, 3, padding=1), wt.ReLU(), wt.MaxPool2d(2),
        wt.Flatten(), wt.Linear(16 * 3 * 3, 2))
    opt = wt.Adam(model.parameters(), lr=3e-3)

    curve = []
    for step in range(60):
        loss = wt.cross_entropy(model(Xt), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 15 == 0 or step == 59:
            curve.append(round(float(loss.item()), 3))

    pred = model(Xt).numpy().argmax(1)
    r["loss_curve"] = curve
    r["accuracy"] = round(float((pred == y).mean()), 3)
    r["ok"] = bool(curve[-1] < curve[0] * 0.6 and r["accuracy"] > 0.8)
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)


main()
