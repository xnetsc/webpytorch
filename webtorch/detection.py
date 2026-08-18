"""webtorch.detection -- object detection (DETR + YOLOv8) in the browser.

Two detectors, both validated to ~1e-5 vs their reference implementation and run
with webtorch's GPU Conv2d:
    det = await detection.DetrDetector.from_npz("/models/detr_web.npz")   # transformer, no NMS
    det = await detection.YoloDetector.from_npz("/models/yolov8n_web.npz")# CNN, DFL + NMS
    results = det.detect(pil_image, threshold=0.25)   # [{name, score, box:[x0,y0,x1,y1]}]

DETR: ResNet-50 backbone + transformer encoder/decoder + set-prediction heads.
YOLOv8n: CSP backbone (Conv/C2f/SPPF) + PAN neck + decoupled DFL head + NMS. All
convolutions run as GPU kernels; the few small deep-layer maxpool/upsample ops and
the head decode/NMS run host-side.
"""
import io, math, json
import numpy as np
from . import _core as wt

xp = wt.xp
STAGES = [3, 4, 6, 3]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
COCO91 = None  # filled from a static list below


def _maxpool_3x3s2p1(x):
    """3x3 stride-2 pad-1 max pool (runs once in the backbone; host numpy)."""
    d = x.numpy()
    N, C, H, W = d.shape
    pad = np.full((N, C, H + 2, W + 2), -1e30, np.float32); pad[:, :, 1:H + 1, 1:W + 1] = d
    OH = (H + 2 - 3) // 2 + 1; OW = (W + 2 - 3) // 2 + 1
    out = np.full((N, C, OH, OW), -1e30, np.float32)
    for i in range(3):
        for j in range(3):
            out = np.maximum(out, pad[:, :, i:i + 2 * OH:2, j:j + 2 * OW:2])
    return wt.Tensor(np.ascontiguousarray(out, np.float32))


class DetrDetector:
    def __init__(self, W):
        self.W = W                       # dict name -> np array (fp32)
        self.H = 256; self.NH = 8; self.HD = 32

    # ---- ResNet-50 backbone (folded BN -> conv w/ bias) ----
    def _conv(self, x, pfx, stride=1, padding=0):
        return wt.conv2d(x, self._t(pfx + ".w"), self._t(pfx + ".b"), stride, padding)

    def _t(self, name):
        t = self._cache.get(name)
        if t is None:
            t = wt.Tensor(self.W[name].astype(np.float32)); self._cache[name] = t
        return t

    def _backbone(self, x):
        x = self._conv(x, "emb", stride=2, padding=3).relu()
        x = _maxpool_3x3s2p1(x)
        for si, nl in enumerate(STAGES):
            for li in range(nl):
                stride = 2 if (si > 0 and li == 0) else 1
                idt = x
                if ("s%d.l%d.sc.w" % (si, li)) in self.W:
                    idt = self._conv(x, "s%d.l%d.sc" % (si, li), stride=stride)
                h = self._conv(x, "s%d.l%d.c0" % (si, li)).relu()
                h = self._conv(h, "s%d.l%d.c1" % (si, li), stride=stride, padding=1).relu()
                h = self._conv(h, "s%d.l%d.c2" % (si, li))
                x = (h + idt).relu()
        return x                          # (1,2048,h,w)

    # ---- sine 2D position embedding (normalize=True, scale 2pi) ----
    def _pos_embed(self, h, w):
        npf = 128; temp = 10000.0; scale = 2 * math.pi; eps = 1e-6
        y = np.arange(1, h + 1, dtype=np.float32)[:, None].repeat(w, 1)
        x = np.arange(1, w + 1, dtype=np.float32)[None, :].repeat(h, 0)
        y = y / (y[-1:, :] + eps) * scale; x = x / (x[:, -1:] + eps) * scale
        dim = np.arange(npf, dtype=np.float32)
        dim = temp ** (2 * (dim // 2) / npf)
        px = x[:, :, None] / dim; py = y[:, :, None] / dim
        px = np.stack([np.sin(px[:, :, 0::2]), np.cos(px[:, :, 1::2])], 3).reshape(h, w, -1)
        py = np.stack([np.sin(py[:, :, 0::2]), np.cos(py[:, :, 1::2])], 3).reshape(h, w, -1)
        pos = np.concatenate([py, px], 2).reshape(h * w, 2 * npf)   # (h*w,256)
        return pos.astype(np.float32)

    # ---- transformer helpers ----
    def _ln(self, x, p):
        w = self._t(p + ".w"); b = self._t(p + ".b")
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return xc / (var + 1e-5).sqrt() * w + b

    def _lin(self, x, p):
        key = p + ".w_T"
        if key not in self._cache:
            self._cache[key] = wt.Tensor(np.ascontiguousarray(self.W[p + ".w"].astype(np.float32).T))
        return x.matmul(self._cache[key]) + self._t(p + ".b")

    def _mha(self, q_in, k_in, v_in, p):
        NH, HD = self.NH, self.HD
        q = self._lin(q_in, p + ".q_proj").reshape(-1, NH, HD).permute(1, 0, 2)
        k = self._lin(k_in, p + ".k_proj").reshape(-1, NH, HD).permute(1, 0, 2)
        v = self._lin(v_in, p + ".v_proj").reshape(-1, NH, HD).permute(1, 0, 2)
        sc = wt.bmm(wt._contig(q), wt.transpose_last2(wt._contig(k))) * (HD ** -0.5)
        a = wt.softmax(sc)
        o = wt.bmm(a, wt._contig(v)).permute(1, 0, 2).reshape(-1, self.H)
        return self._lin(wt._contig(o), p + ".o_proj")

    def forward(self, pixel_values):
        feat = self._backbone(wt.Tensor(pixel_values))       # (1,2048,h,w)
        _, _, h, w = feat.shape
        src = self._conv(feat, "inproj")                     # 1x1 conv -> (1,256,h,w)
        src = wt._contig(src.reshape(self.H, h * w).permute(1, 0))  # (h*w,256)
        spos = wt.Tensor(self._pos_embed(h, w))
        x = src
        for i in range(6):
            p = "encoder.%d" % i; qk = x + spos
            x = self._ln(x + self._mha(qk, qk, x, p + ".self_attn"), p + ".self_attn_layer_norm")
            m = self._lin(self._lin(x, p + ".mlp.fc1").relu(), p + ".mlp.fc2")
            x = self._ln(x + m, p + ".final_layer_norm")
        memory = x
        qpos = self._t("qpos")
        t = wt.Tensor(np.zeros((qpos.shape[0], self.H), np.float32))
        for i in range(6):
            p = "decoder.%d" % i; qk = t + qpos
            t = self._ln(t + self._mha(qk, qk, t, p + ".self_attn"), p + ".self_attn_layer_norm")
            ca = self._mha(t + qpos, memory + spos, memory, p + ".encoder_attn")
            t = self._ln(t + ca, p + ".encoder_attn_layer_norm")
            m = self._lin(self._lin(t, p + ".mlp.fc1").relu(), p + ".mlp.fc2")
            t = self._ln(t + m, p + ".final_layer_norm")
        t = self._ln(t, "dec.ln")
        logits = self._lin(t, "cls")
        b = self._lin(t, "bbox.0").relu(); b = self._lin(b, "bbox.1").relu()
        boxes = self._lin(b, "bbox.2").sigmoid()
        return logits.numpy(), boxes.numpy()

    # ---- image pre/post ----
    def preprocess(self, pil_img, shortest=320, longest=480):
        from PIL import Image
        img = pil_img.convert("RGB"); W0, H0 = img.size
        s = shortest / min(H0, W0)
        if max(H0, W0) * s > longest:
            s = longest / max(H0, W0)
        Wn, Hn = round(W0 * s), round(H0 * s)
        img = img.resize((Wn, Hn), Image.BILINEAR)
        a = np.asarray(img, np.float32) / 255.0
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        a = a.transpose(2, 0, 1)[None]                       # (1,3,H,W)
        return np.ascontiguousarray(a, np.float32), (H0, W0)

    def detect(self, pil_img, threshold=0.7):
        pv, (H0, W0) = self.preprocess(pil_img)
        logits, boxes = self.forward(pv)
        e = np.exp(logits - logits.max(-1, keepdims=True)); prob = e / e.sum(-1, keepdims=True)
        cls = prob[:, :-1]; score = cls.max(-1); label = cls.argmax(-1)
        keep = score > threshold
        out = []
        for i in np.where(keep)[0]:
            cx, cy, bw, bh = boxes[i]
            x0 = (cx - bw / 2) * W0; y0 = (cy - bh / 2) * H0
            x1 = (cx + bw / 2) * W0; y1 = (cy + bh / 2) * H0
            out.append({"label": int(label[i]), "name": COCO91[int(label[i])] if COCO91 else str(int(label[i])),
                        "score": round(float(score[i]), 3),
                        "box": [round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1)]})
        return out

    @classmethod
    async def from_npz(cls, url):
        from . import webio
        data = await webio.load_npz(url)
        W = {k: data[k] for k in data}
        self = cls(W); self._cache = {}
        return self


COCO91 = ["N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
          "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
          "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
          "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"]


# ============================ YOLOv8 ==========================================
class YoloDetector:
    """YOLOv8n: BN-fused CSP backbone + PAN neck + DFL head, GPU convs + host NMS."""
    def __init__(self, W, names):
        self.W = W; self.names = names; self._cache = {}
        self.strides = [8.0, 16.0, 32.0]; self.nc = 80; self.reg_max = 16

    def _t(self, name):
        t = self._cache.get(name)
        if t is None:
            t = wt.Tensor(self.W[name].astype(np.float32)); self._cache[name] = t
        return t

    def _cv(self, x, pfx, stride=1):                     # Conv = SiLU(conv), pad k//2
        w = self._t(pfx + ".conv.weight"); k = self.W[pfx + ".conv.weight"].shape[2]
        return wt.silu(wt.conv2d(x, w, self._t(pfx + ".conv.bias"), stride, k // 2))

    def _plain(self, x, pfx):                            # 1x1 conv, no activation
        return wt.conv2d(x, self._t(pfx + ".weight"), self._t(pfx + ".bias"), 1, 0)

    def _bottleneck(self, x, pfx, shortcut):
        y = self._cv(self._cv(x, pfx + ".cv1"), pfx + ".cv2")
        return (x + y) if shortcut else y

    def _c2f(self, x, pfx, n, shortcut=True):
        y = self._cv(x, pfx + ".cv1"); c = y.shape[1] // 2
        a = wt._contig(y.data[:, :c]); b = wt._contig(y.data[:, c:])
        a = wt.Tensor(a); cur = wt.Tensor(b); outs = [a, cur]
        for i in range(n):
            cur = self._bottleneck(cur, f"{pfx}.m.{i}", shortcut); outs.append(cur)
        return self._cv(wt.cat(outs, axis=1), pfx + ".cv2")

    def _maxpool5(self, x):                              # 5x5 s1 p2 (host; small)
        d = x.numpy(); N, C, H, W = d.shape
        pad = np.pad(d, ((0, 0), (0, 0), (2, 2), (2, 2)), constant_values=-1e30)
        out = np.full((N, C, H, W), -1e30, np.float32)
        for i in range(5):
            for j in range(5):
                out = np.maximum(out, pad[:, :, i:i + H, j:j + W])
        return wt.Tensor(np.ascontiguousarray(out))

    def _sppf(self, x, pfx):
        x = self._cv(x, pfx + ".cv1"); p1 = self._maxpool5(x); p2 = self._maxpool5(p1); p3 = self._maxpool5(p2)
        return self._cv(wt.cat([x, p1, p2, p3], axis=1), pfx + ".cv2")

    def _upsample(self, x):                              # nearest 2x (host; deep/small)
        d = x.numpy()
        return wt.Tensor(np.ascontiguousarray(d.repeat(2, 2).repeat(2, 3)))

    def forward(self, pv):
        x = wt.Tensor(pv)
        o = {}
        o[0] = self._cv(x, "model.0", 2); o[1] = self._cv(o[0], "model.1", 2); o[2] = self._c2f(o[1], "model.2", 1)
        o[3] = self._cv(o[2], "model.3", 2); o[4] = self._c2f(o[3], "model.4", 2)
        o[5] = self._cv(o[4], "model.5", 2); o[6] = self._c2f(o[5], "model.6", 2)
        o[7] = self._cv(o[6], "model.7", 2); o[8] = self._c2f(o[7], "model.8", 1); o[9] = self._sppf(o[8], "model.9")
        o[10] = self._upsample(o[9]); o[11] = wt.cat([o[10], o[6]], axis=1); o[12] = self._c2f(o[11], "model.12", 1, False)
        o[13] = self._upsample(o[12]); o[14] = wt.cat([o[13], o[4]], axis=1); o[15] = self._c2f(o[14], "model.15", 1, False)
        o[16] = self._cv(o[15], "model.16", 2); o[17] = wt.cat([o[16], o[12]], axis=1); o[18] = self._c2f(o[17], "model.18", 1, False)
        o[19] = self._cv(o[18], "model.19", 2); o[20] = wt.cat([o[19], o[9]], axis=1); o[21] = self._c2f(o[20], "model.21", 1, False)
        # Detect head (host decode)
        dflw = self.W["model.22.dfl.conv.weight"][0, :, 0, 0].astype(np.float32)
        boxes = []; clses = []
        for si, f in enumerate([o[15], o[18], o[21]]):
            box = self._cv(f, f"model.22.cv2.{si}.0"); box = self._cv(box, f"model.22.cv2.{si}.1")
            box = self._plain(box, f"model.22.cv2.{si}.2").numpy()[0]        # (64,H,W)
            cls = self._cv(f, f"model.22.cv3.{si}.0"); cls = self._cv(cls, f"model.22.cv3.{si}.1")
            cls = self._plain(cls, f"model.22.cv3.{si}.2").numpy()[0]        # (80,H,W)
            C, H, W = box.shape; box = box.reshape(4, 16, H * W)
            e = np.exp(box - box.max(1, keepdims=True)); sm = e / e.sum(1, keepdims=True)
            dist = (sm * dflw[None, :, None]).sum(1)                         # (4,HW) ltrb
            gx, gy = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
            ap = np.stack([gx.reshape(-1), gy.reshape(-1)], 0)
            x1y1 = ap - dist[:2]; x2y2 = ap + dist[2:]
            dbox = np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], 0) * self.strides[si]
            boxes.append(dbox); clses.append(1.0 / (1.0 + np.exp(-cls.reshape(self.nc, H * W))))
        return np.concatenate(boxes, 1), np.concatenate(clses, 1)            # (4,6300),(80,6300)

    @staticmethod
    def _nms(xywh, conf, cls, iou_t):
        x1 = xywh[:, 0] - xywh[:, 2] / 2; y1 = xywh[:, 1] - xywh[:, 3] / 2
        x2 = xywh[:, 0] + xywh[:, 2] / 2; y2 = xywh[:, 1] + xywh[:, 3] / 2
        xy = np.stack([x1, y1, x2, y2], 1); out = []
        for c in np.unique(cls):
            m = cls == c; bx = xy[m]; sc = conf[m]; order = sc.argsort()[::-1]
            while len(order):
                i = order[0]; out.append((bx[i], sc[i], int(c)))
                if len(order) == 1: break
                rest = order[1:]
                xx1 = np.maximum(bx[i, 0], bx[rest, 0]); yy1 = np.maximum(bx[i, 1], bx[rest, 1])
                xx2 = np.minimum(bx[i, 2], bx[rest, 2]); yy2 = np.minimum(bx[i, 3], bx[rest, 3])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                a1 = (bx[i, 2] - bx[i, 0]) * (bx[i, 3] - bx[i, 1]); a2 = (bx[rest, 2] - bx[rest, 0]) * (bx[rest, 3] - bx[rest, 1])
                iou = inter / (a1 + a2 - inter + 1e-9); order = rest[iou < iou_t]
        return out

    def preprocess(self, pil_img, imgsz=640):
        from PIL import Image
        img = pil_img.convert("RGB"); W0, H0 = img.size
        r = min(imgsz / H0, imgsz / W0)
        img2 = img.resize((round(W0 * r), round(H0 * r)), Image.BILINEAR)
        nw, nh = img2.size
        padw = (32 - nw % 32) % 32; padh = (32 - nh % 32) % 32  # pad bottom/right to /32
        canvas = np.full((nh + padh, nw + padw, 3), 114, np.uint8)
        canvas[:nh, :nw] = np.asarray(img2)
        a = canvas.astype(np.float32) / 255.0
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None], np.float32), r

    def detect(self, pil_img, threshold=0.25, iou_t=0.7, imgsz=640):
        pv, r = self.preprocess(pil_img, imgsz)
        box, cls = self.forward(pv)
        conf = cls.max(0); lab = cls.argmax(0); keep = conf > threshold
        dets = self._nms(box.T[keep], conf[keep], lab[keep], iou_t)
        out = []
        for bx, sc, c in sorted(dets, key=lambda t: -t[1]):
            xyxy = [float(v) / r for v in bx]
            out.append({"label": c, "name": self.names.get(str(c), self.names.get(c, str(c))),
                        "score": round(float(sc), 3), "box": [round(v, 1) for v in xyxy]})
        return out

    @classmethod
    async def from_npz(cls, url, names_url=None):
        from . import webio
        data = await webio.load_npz(url)
        W = {k: data[k] for k in data}
        names = await webio.read_json(names_url) if names_url else {}
        return cls(W, names)
