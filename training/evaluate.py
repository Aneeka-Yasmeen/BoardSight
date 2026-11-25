import argparse
import collections
import json

import cv2
import numpy as np
from ultralytics import YOLO

from common import CLASSES, iou, phone_sim, quad

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--long-side", type=int, default=2300)
ap.add_argument("--iou", type=float, default=0.33)
ap.add_argument("--phone", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sets", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()
model = YOLO(a.weights, task="detect")
names = model.names
TH = [0.15, 0.25, 0.35, 0.45, 0.55]


def infer(imgs):
    out = []
    for k in range(0, len(imgs), 16):
        for r in model.predict(imgs[k:k + 16], imgsz=640, conf=0.10, iou=0.5, verbose=False):
            out.append([(CLASSES.index(names[int(b.cls)]), float(b.conf), *map(float, b.xyxy[0])) for b in r.boxes if names[int(b.cls)] in CLASSES])
    return out


def nms(dets, thr=0.45):
    dets = sorted(dets, key=lambda x: -x[1])
    keep = []
    for d in dets:
        if all(iou(d[2:], k[2:]) < thr for k in keep):
            keep.append(d)
    return keep


def tiled(img, long_side):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    H, W = im.shape[:2]
    pos = lambda n: [0] if n <= 640 else [int(round(v)) for v in np.linspace(0, n - 640, int(np.ceil((n - 640) / 512)) + 1)]
    tiles, off = [], []
    for y in pos(H):
        for x in pos(W):
            t = np.full((640, 640, 3), 114, np.uint8)
            c = im[y:y + 640, x:x + 640]
            t[: c.shape[0], : c.shape[1]] = c
            tiles.append(t)
            off.append((x, y))
    out = []
    for (x, y), ds in zip(off, infer(tiles)):
        for c, p, x1, y1, x2, y2 in ds:
            out.append((c, p, (x1 + x) / s, (y1 + y) / s, (x2 + x) / s, (y2 + y) / s))
    return nms(out)


def match(gt, det):
    used, tp, loc = set(), 0, 0
    per = collections.Counter()
    for d in sorted(det, key=lambda x: -x[1]):
        best, bi = 0, -1
        for i, g in enumerate(gt):
            if i not in used and g[0] == d[0]:
                v = iou(d[2:], g[1:])
                if v > best:
                    best, bi = v, i
        if best >= a.iou:
            used.add(bi)
            tp += 1
            per[("tp", d[0])] += 1
        else:
            per[("fp", d[0])] += 1
    for i, g in enumerate(gt):
        per[("gt", g[0])] += 1
        if any(iou(d[2:], g[1:]) >= a.iou for d in det):
            loc += 1
    return tp, len(det) - tp, len(gt) - tp, loc, per


tests = json.load(open(a.tests))
report = {}
for name in sorted({t["set"] for t in tests}):
    if a.sets and name not in a.sets.split(","):
        continue
    items = [t for t in tests if t["set"] == name]
    if a.limit and len(items) > a.limit:
        items = items[:: max(1, len(items) // a.limit)][: a.limit]
    if name == "dspcb":
        items = [{"set": name, "mode": "full", "group": items[k:k + 4], "boxes": []} for k in range(0, len(items) - 3, 4)]
    rng = np.random.default_rng(123)
    acc = {t: [0, 0, 0, 0] for t in TH}
    per = collections.Counter()
    clean = {t: 0 for t in TH}
    gts = 0
    for t in items:
        if "group" in t:
            img, boxes = quad([(cv2.imread(g["image"]), g["boxes"]) for g in t["group"]])
        else:
            img = cv2.imread(t["image"])
            boxes = [tuple(b) for b in t["boxes"]]
        if a.phone:
            img, boxes = phone_sim(img, boxes, rng, 0.8)
        det = tiled(img, a.long_side) if t["mode"] == "tile" else nms(infer([img])[0])
        valid = {CLASSES.index(n) for n in names.values() if n in CLASSES}
        gt = [b for b in boxes if b[0] in valid]
        gts += len(gt)
        for th in TH:
            dd = [x for x in det if x[1] >= th]
            r = match(gt, dd)
            for k in range(4):
                acc[th][k] += r[k]
            clean[th] += bool(dd)
            if th == 0.25:
                per += r[4]
    report[name] = {"images": len(items), "gt": gts}
    if not gts:
        print(f"{name}: {len(items)} clean images; images with any false alarm: " + ", ".join(f"@{t} {clean[t] / len(items):.1%}" for t in TH))
        report[name]["false_alarm"] = {str(t): clean[t] / len(items) for t in TH}
        continue
    print(f"{name}: {len(items)} images, {gts} defects (IoU>{a.iou})")
    report[name]["at"] = {}
    for t in TH:
        tp, fp, fn, loc = acc[t]
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-9, p + r)
        report[name]["at"][str(t)] = {"p": p, "r": r, "f1": f, "located": loc / gts}
        print(f"   conf {t}: P {p:.3f} R {r:.3f} F1 {f:.3f} | found at all (any type) {loc / gts:.3f}")
    print("   per class @0.25: " + ", ".join(f"{CLASSES[c]} P{per[('tp', c)] / max(1, per[('tp', c)] + per[('fp', c)]):.2f}/R{per[('tp', c)] / max(1, per[('gt', c)]):.2f}" for c in range(len(CLASSES)) if per[("gt", c)]))
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
