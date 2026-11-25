import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "desktop"))
from common import CLASSES, iou, quad
from pcbqc import detector

ap = argparse.ArgumentParser()
ap.add_argument("--tests", required=True)
ap.add_argument("--limit", type=int, default=250)
ap.add_argument("--iou", type=float, default=0.33)
a = ap.parse_args()
tests = json.load(open(a.tests))
for name in ("pcbdefect", "pku", "deeppcb", "deeppcb_clean", "dspcb"):
    items = [t for t in tests if t["set"] == name]
    items = items[:: max(1, len(items) // a.limit)][: a.limit]
    if name == "dspcb":
        items = [items[k:k + 4] for k in range(0, len(items) - 3, 4)]
    tp = fp = fn = flagged = 0
    for t in items:
        if name == "dspcb":
            img, boxes = quad([(cv2.imread(g["image"]), g["boxes"]) for g in t])
        else:
            img, boxes = cv2.imread(t["image"]), t["boxes"]
        dets = detector.detect(img)
        flagged += bool(dets)
        used = set()
        for d in sorted(dets, key=lambda v: -v["conf"]):
            x, y, w, h = d["bbox"]
            options = [(iou((x, y, x + w, y + h), g[1:]), j) for j, g in enumerate(boxes) if j not in used and CLASSES[int(g[0])] == d["cls"]]
            best = max(options, default=(0, -1))
            if best[0] >= a.iou:
                used.add(best[1])
                tp += 1
            else:
                fp += 1
        fn += len(boxes) - len(used)
    if name == "deeppcb_clean":
        print(f"{name}: {len(items)} images, {flagged / len(items):.1%} with a false alarm")
    else:
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        print(f"{name}: {len(items)} images, precision {p:.3f}, recall {r:.3f}, F1 {2 * p * r / max(1e-9, p + r):.3f}")
