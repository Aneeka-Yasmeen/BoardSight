import json
import os

import cv2
import numpy as np
import onnxruntime as ort

SIZE, STRIDE, LONG_SIDE = 640, 560, 2300
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.environ.get("PCB_ASSETS", os.path.join(ROOT, "android", "app", "src", "main", "assets"))
_session = None
_classes = []


def load():
    global _session, _classes
    if _session is None:
        _session = ort.InferenceSession(os.path.join(ASSETS, "pcb_yolo.onnx"), providers=["CPUExecutionProvider"])
        with open(os.path.join(ASSETS, "classes.json")) as f:
            _classes = json.load(f)
    return _session


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def nms(d, thr):
    keep = []
    for x in sorted(d, key=lambda v: -v[1]):
        if all(iou(x[2:], k[2:]) < thr for k in keep):
            keep.append(x)
    return keep


def _tile(img, conf):
    s = load()
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    p = s.run(None, {s.get_inputs()[0].name: x})[0][0]
    cls, score = p[4:].argmax(0), p[4:].max(0)
    out = []
    for i in np.where(score >= conf)[0]:
        cx, cy, w, h = (float(v) for v in p[:4, i])
        out.append((int(cls[i]), float(score[i]), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return nms(out, 0.5)


def positions(n):
    if n <= SIZE:
        return [0]
    return [int(round(v)) for v in np.linspace(0, n - SIZE, int(np.ceil((n - SIZE) / STRIDE)) + 1)]


def fuse(dets):
    def shared(a, b):
        w = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
        h = max(0.0, min(a[5], b[5]) - max(a[3], b[3]))
        return w * h / max(1e-6, min((a[4] - a[2]) * (a[5] - a[3]), (b[4] - b[2]) * (b[5] - b[3])))

    electrical = {_classes.index(c) for c in ("short", "open_circuit")}
    keep = []
    for d in sorted(dets, key=lambda v: -v[1]):
        same = next((i for i, k in enumerate(keep) if k[0] == d[0] and shared(k, d) > 0.5), None)
        if same is not None:
            k = keep[same]
            keep[same] = (k[0], k[1], min(k[2], d[2]), min(k[3], d[3]), max(k[4], d[4]), max(k[5], d[5]))
        elif not any(shared(k, d) > 0.7 and (d[0] not in electrical or k[0] in electrical) for k in keep):
            keep.append(d)
    return keep


def detect(img, conf=0.45):
    big = max(img.shape[:2])
    s = SIZE / big if big <= 900 else LONG_SIDE / big
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    found = []
    for y in positions(im.shape[0]):
        for x in positions(im.shape[1]):
            t = np.full((SIZE, SIZE, 3), 114, np.uint8)
            c = im[y:y + SIZE, x:x + SIZE]
            t[: c.shape[0], : c.shape[1]] = c
            found += [(k, p, (x1 + x) / s, (y1 + y) / s, (x2 + x) / s, (y2 + y) / s) for k, p, x1, y1, x2, y2 in _tile(t, conf)]
    return [{"cls": _classes[k], "conf": p, "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], "source": "yolo"}
            for k, p, x1, y1, x2, y2 in fuse(nms(found, 0.45))]
