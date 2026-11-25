import argparse
import glob
import json
import os
import sys

import cv2
import fitz
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pku", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
names = open(os.path.join(a.pku, "classes.txt")).read().split()
CREDIT = "PKU-Market-PCB, Huang & Wei, arXiv:1901.08204. Real board photo; defects were edited in by the dataset authors. Not used in training."
pages, key = [], {}


def load(f):
    img = cv2.imread(f)
    h, w = img.shape[:2]
    boxes = []
    for row in open(f.replace("images", "labels")[:-4] + ".txt"):
        c, x, y, bw, bh = map(float, row.split()[:5])
        boxes.append([names[int(c)], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return img, boxes


def add(name, img, boxes, title):
    cv2.imwrite(os.path.join(a.out, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    key[name] = boxes
    pages.append((name, title))


for k, board in enumerate(("10", "11", "12")):
    files = sorted(glob.glob(os.path.join(a.pku, "images", board + "_*.jpg")))
    pick = files[:: max(1, len(files) // 11)][:11]
    add(f"board{board}_reference", np.median(np.stack([cv2.imread(f) for f in pick]), axis=0).astype(np.uint8), [], "clean reference, median of photos of this board")
    for cls in (names[k], names[k + 3]):
        f = next(x for x in files if "_" + cls + "_" in os.path.basename(x))
        img, boxes = load(f)
        add(f"board{board}_{cls}", img, boxes, cls.replace("_", " "))
json.dump(key, open(os.path.join(a.out, "answers.json"), "w"))
doc = fitz.open()
for name, title in pages:
    p = os.path.join(a.out, name + ".jpg")
    h, w = cv2.imread(p).shape[:2]
    pw, ph = (842, 595) if w > h else (595, 842)
    page = doc.new_page(width=pw, height=ph)
    page.insert_image(fitz.Rect(28, 24, pw - 28, ph - 46), filename=p, keep_proportion=True)
    page.insert_text((28, ph - 28), f"{name}: {title}", fontsize=8)
    page.insert_text((28, ph - 16), CREDIT, fontsize=7)
doc.save(os.path.join(a.out, "pku_photos.pdf"), deflate=True, garbage=3)
print(len(pages), "pages", round(os.path.getsize(os.path.join(a.out, "pku_photos.pdf")) / 1e6, 1), "MB", file=sys.stdout)
