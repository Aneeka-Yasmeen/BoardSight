import argparse
import json
import os

import cv2
import fitz
import numpy as np

from common import CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--tests", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boards", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tests = json.load(open(a.tests))
deep = [t for t in tests if t["set"] == "deeppcb" and len(t["boxes"]) >= 4]
deep = deep[:: max(1, len(deep) // 6)][:6]
S, GAP = 640, 24
W, H = 3 * S + 4 * GAP, 2 * S + 3 * GAP
sheets, key = [], {}


def origin(i):
    return GAP + (i % 3) * (S + GAP), GAP + (i // 3) * (S + GAP)


def panel(kind):
    can = np.full((H, W, 3), 255, np.uint8)
    for i, t in enumerate(deep):
        x, y = origin(i)
        can[y:y + S, x:x + S] = cv2.imread(t["image"].replace("_test", "_temp") if kind == "clean" else t["image"])
    return can


def patch(can, i, j):
    c, x1, y1, x2, y2 = [int(v) for v in deep[i]["boxes"][j]]
    x1, y1, x2, y2 = max(0, x1 - 6), max(0, y1 - 6), min(S, x2 + 6), min(S, y2 + 6)
    ox, oy = origin(i)
    can[oy + y1:oy + y2, ox + x1:ox + x2] = cv2.imread(deep[i]["image"])[y1:y2, x1:x2]
    return [CLASSES[c], ox + x1, oy + y1, ox + x2, oy + y2]


def add(name, img, note, boxes):
    p = os.path.join(a.out, name + ".png")
    cv2.imwrite(p, img)
    sheets.append((p, f"{name}: {note}"))
    key[name] = boxes


clean = panel("clean")
add("1_reference", clean, "clean reference panel. Set this one as the reference board.", [])
hot = max(
    ((i, j) for i, t in enumerate(deep) for j, b in enumerate(t["boxes"]) if CLASSES[int(b[0])] in ("short", "open_circuit")),
    key=lambda ij: (deep[ij[0]]["boxes"][ij[1]][3] - deep[ij[0]]["boxes"][ij[1]][1]) * (deep[ij[0]]["boxes"][ij[1]][4] - deep[ij[0]]["boxes"][ij[1]][2]),
)
for k in range(4):
    img = clean.copy()
    boxes = [patch(img, *hot)]
    i = (hot[0] + 1 + k) % 6
    boxes.append(patch(img, i, k % len(deep[i]["boxes"])))
    add(f"2_mask_fault_{k + 1}", img, "simulated mask fault, real defect placed at the same spot on every sheet", boxes)
full = []
for i, t in enumerate(deep):
    ox, oy = origin(i)
    full += [[CLASSES[int(c)], ox + x1, oy + y1, ox + x2, oy + y2] for c, x1, y1, x2, y2 in t["boxes"]]
add("3_many_defects", panel("test"), "six untouched DeepPCB test images", full)
pd = [t for t in tests if t["set"] == "pcbdefect"]
for k, t in enumerate(pd[:: max(1, len(pd) // a.boards)][: a.boards]):
    img = cv2.imread(t["image"])
    s = 2300 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    add(f"4_etched_board_{k + 1}", img, "real etched board scan", [[CLASSES[int(c)], x1 * s, y1 * s, x2 * s, y2 * s] for c, x1, y1, x2, y2 in t["boxes"]])
json.dump(key, open(os.path.join(a.out, "answers.json"), "w"))
doc = fitz.open()
for p, title in sheets:
    h, w = cv2.imread(p).shape[:2]
    pw, ph = (842, 595) if w > h else (595, 842)
    page = doc.new_page(width=pw, height=ph)
    page.insert_image(fitz.Rect(28, 24, pw - 28, ph - 46), stream=cv2.imencode(".jpg", cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), keep_proportion=True)
    page.insert_text((28, ph - 28), title, fontsize=8)
    src = "PCB-Defect, Mendeley Data 10.17632/vdj74sngvn.1, CC BY 4.0, resized" if "etched" in title else "DeepPCB, github.com/tangsanli5201/DeepPCB, MIT"
    page.insert_text((28, ph - 16), src + ". Held-out test images.", fontsize=7)
doc.save(os.path.join(a.out, "demo_sheets.pdf"), deflate=True, garbage=3)
print(len(sheets), "sheets", round(os.path.getsize(os.path.join(a.out, "demo_sheets.pdf")) / 1e6, 1), "MB")
