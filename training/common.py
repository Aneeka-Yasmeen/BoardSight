import cv2
import numpy as np

CLASSES = ["open_circuit", "short", "mouse_bite", "spur", "spurious_copper", "pin_hole", "missing_hole", "missing_pad"]
CID = {n: i for i, n in enumerate(CLASSES)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_yolo(boxes, w, h):
    out = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return "\n".join(out)


def warp(img, boxes, rng, strength):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-strength, strength, (4, 2)).astype(np.float32) * np.float32([w, h])
    m = cv2.getPerspectiveTransform(src, dst)
    res = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = []
    for c, x1, y1, x2, y2 in boxes:
        p = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), m)[0]
        out.append((c, float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())))
    return res, out


def phone_sim(img, boxes, rng, level=1.0):
    h, w = img.shape[:2]
    if rng.random() < 0.7:
        img, boxes = warp(img, boxes, rng, 0.03 * level)
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if rng.random() < 0.7:
        a = rng.uniform(0, 2 * np.pi)
        g = (np.cos(a) * (xx / w - 0.5) + np.sin(a) * (yy / h - 0.5)) * rng.uniform(0.15, 0.7) * level
        f *= (1 + g)[..., None]
    if rng.random() < 0.6:
        f *= rng.uniform(1 - 0.16 * level, 1 + 0.16 * level, 3).astype(np.float32)
    if rng.random() < 0.3 * level:
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.06, 0.22) * w
        f += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(40, 130))[..., None]
    if rng.random() < 0.5:
        f = 255 * np.power(np.clip(f, 0, 255) / 255, rng.uniform(0.75, 1.35))
    img = np.clip(f, 0, 255).astype(np.uint8)
    p = rng.random()
    if p < 0.30:
        img = cv2.GaussianBlur(img, (0, 0), max(0.3, rng.uniform(0.5, 1.5) * level))
    elif p < 0.45:
        n = int(rng.integers(3, 8))
        k = np.zeros((n, n), np.float32)
        k[n // 2] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1), (n, n))
        img = cv2.filter2D(img, -1, k / max(float(k.sum()), 1e-6))
    elif p < 0.65:
        s = rng.uniform(0.5, 0.85)
        img = cv2.resize(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.12:
        a, t = rng.uniform(0, np.pi), rng.uniform(2.6, 6)
        img = np.clip(img + (np.sin((np.cos(a) * xx + np.sin(a) * yy) * 2 * np.pi / t) * rng.uniform(3, 10))[..., None], 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        img = np.clip(img + rng.normal(0, rng.uniform(1, 7) * level, img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 92))])[1], 1)
    return img, boxes


def colourise(gray, rng):
    h, w = gray.shape
    soft = cv2.GaussianBlur(gray, (0, 0), rng.uniform(0.5, 1.1)).astype(np.float32) / 255
    schemes = [((30, 70, 25), (70, 140, 60)), ((25, 25, 22), (120, 160, 205)), ((60, 130, 140), (55, 75, 120)),
               ((70, 35, 20), (150, 110, 60)), ((35, 80, 30), (110, 190, 205)), ((20, 20, 70), (70, 80, 170))]
    sub, cop = schemes[int(rng.integers(len(schemes)))]
    sub = np.float32(sub) * rng.uniform(0.75, 1.3)
    cop = np.float32(cop) * rng.uniform(0.8, 1.2)
    tex = cv2.resize(rng.normal(0, 1, (h // 8, w // 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    m = soft[..., None]
    out = sub * m + cop * (1 - m)
    out = out * (1 + 0.05 * tex[..., None]) + fine[..., None] * 3
    return np.clip(out, 0, 255).astype(np.uint8)


def quad(items, size=640):
    can = np.zeros((size, size, 3), np.uint8)
    half, boxes = size // 2, []
    for j, (img, bx) in enumerate(items[:4]):
        h, w = img.shape[:2]
        ox, oy = (j % 2) * half, (j // 2) * half
        can[oy:oy + half, ox:ox + half] = cv2.resize(img, (half, half), interpolation=cv2.INTER_CUBIC)
        boxes += [(c, ox + x1 * half / w, oy + y1 * half / h, ox + x2 * half / w, oy + y2 * half / h) for c, x1, y1, x2, y2 in bx]
    return can, boxes
