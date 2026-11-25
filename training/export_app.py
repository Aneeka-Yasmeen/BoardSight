import argparse
import collections
import json
import os
import shutil

import cv2
from ultralytics import YOLO

from common import CLASSES, quad

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--assets", required=True)
a = ap.parse_args()

os.makedirs(os.path.join(a.assets, "bench"), exist_ok=True)
onnx = YOLO(a.weights).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
shutil.copy(onnx, os.path.join(a.assets, "pcb_yolo.onnx"))
names = YOLO(a.weights).names
assert [names[i] for i in range(len(names))] == CLASSES
json.dump(CLASSES, open(os.path.join(a.assets, "classes.json"), "w"))

tests = json.load(open(a.tests))
want = {"pcbdefect": 24, "deeppcb": 50, "deeppcb_clean": 30, "dspcb": 80}
by = collections.defaultdict(list)
for t in tests:
    if t["set"] in want:
        by[t["set"]].append(t)
out = []
for s, n in want.items():
    items = by[s]
    step = max(1, len(items) // n)
    chosen = items[::step][:n]
    if s == "dspcb":
        chosen = [{"quad": chosen[k:k + 4], "mode": "full"} for k in range(0, len(chosen) - 3, 4)]
    for k, t in enumerate(chosen):
        if "quad" in t:
            img, boxes = quad([(cv2.imread(q["image"]), q["boxes"]) for q in t["quad"]])
        else:
            img, boxes = cv2.imread(t["image"]), t["boxes"]
        h, w = img.shape[:2]
        f = 2300 / max(h, w) if t["mode"] == "tile" else 1.0
        if f != 1.0:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        name = f"{s}_{k:03d}.jpg"
        cv2.imwrite(os.path.join(a.assets, "bench", name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append({"file": name, "set": s, "boxes": [[CLASSES[int(c)], round(x1 * f, 1), round(y1 * f, 1), round(x2 * f, 1), round(y2 * f, 1)] for c, x1, y1, x2, y2 in boxes]})
json.dump(out, open(os.path.join(a.assets, "bench", "bench.json"), "w"))
print(len(out), "bench images,", round(sum(os.path.getsize(os.path.join(a.assets, "bench", o["file"])) for o in out) / 1e6, 1), "MB")
