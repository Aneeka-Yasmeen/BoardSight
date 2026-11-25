import os
import time

import cv2

from . import advisor, detector, logdb, reference, severity

COLORS = {"critical": (53, 57, 229), "major": (0, 124, 239), "minor": (0, 148, 190)}


def quality(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = min(1.0, 1200 / max(g.shape))
    h = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    p = []
    if cv2.Laplacian(h, cv2.CV_64F).var() < 40:
        p.append("image is blurred")
    if g.mean() < 35:
        p.append("image is too dark")
    if g.mean() > 225:
        p.append("image is too bright")
    return p


def _inter(a, b):
    return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))


def merge(model, golden):
    def explained(g):
        return any(_inter(m["bbox"], g["bbox"]) > 0.3 * min(m["bbox"][2] * m["bbox"][3], g["bbox"][2] * g["bbox"][3]) for m in model)

    return model + [g for g in golden if not explained(g)]


def inspect(img, gold=None, board_id="board", out_dir="out", db="inspections.db", conf=0.45):
    t0 = time.time()
    h, w = img.shape[:2]
    problems = quality(img)
    found = detector.detect(img, conf)
    note = "no reference board set"
    if gold is not None:
        boxes, note, _ = reference.compare(gold, img)
        found = merge(found, [{"cls": "anomaly", "conf": 0.6, "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], "source": "reference"} for x1, y1, x2, y2 in boxes])
    graded = severity.grade(found, w, h)
    v = severity.verdict(graded, problems)
    vision = (time.time() - t0) * 1000
    t1 = time.time()
    advice, used = advisor.explain({"board_id": board_id, "verdict": v, "defects": graded})
    advice_ms = (time.time() - t1) * 1000
    vis = img.copy()
    k = max(1, max(h, w) // 700)
    for d in graded:
        x, y, bw, bh = d["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), COLORS[d["severity"]], k * 2)
        cv2.putText(vis, d["cls"].replace("_", " "), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, COLORS[d["severity"]], k)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{board_id}_{v}.png")
    cv2.imwrite(path, vis)
    logdb.log(db, board_id, v, graded, vision, path, advice)
    return {"verdict": v, "defects": graded, "advice": advice, "llama": used, "reference": note, "problems": problems,
            "vision_ms": round(vision, 1), "advice_ms": round(advice_ms, 1), "image": path}
