import argparse
import collections
import glob
import itertools
import os

import cv2
import numpy as np

import reference
from common import iou, phone_sim

ap = argparse.ArgumentParser()
ap.add_argument("--pku", required=True)
ap.add_argument("--boards", default="01,04,05")
ap.add_argument("--per-board", type=int, default=12)
ap.add_argument("--level", type=float, default=0.8)
ap.add_argument("--grid", action="store_true")
a = ap.parse_args()
cases = []
rng = np.random.default_rng(5)
for b in a.boards.split(","):
    fs = sorted(glob.glob(os.path.join(a.pku, "images", b + "_*.jpg")))
    fs = [f for f in fs if os.path.exists(f.replace("images", "labels")[:-4] + ".txt")]
    if len(fs) < 9:
        print("board", b, "has only", len(fs), "images; skipped")
        continue
    pick = fs[:: max(1, len(fs) // 11)][:11]
    ref = np.median(np.stack([cv2.imread(f) for f in pick]), axis=0).astype(np.uint8)
    for f in fs[:: max(1, len(fs) // a.per_board)][: a.per_board]:
        im = cv2.imread(f)
        h, w = im.shape[:2]
        bx = []
        for row in open(f.replace("images", "labels")[:-4] + ".txt"):
            c, x, y, bw, bh = map(float, row.split()[:5])
            bx.append((int(c), (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
        if a.level > 0:
            im, bx = phone_sim(im, bx, rng, a.level)
        cases.append((b, ref, im, bx))
    r2 = ref if a.level == 0 else phone_sim(ref, [], rng, a.level)[0]
    cases.append((b + "_clean", ref, r2, []))
    if a.level > 0:
        for _ in range(3):
            cases.append((b + "_clean", ref, phone_sim(ref, [], rng, a.level)[0], []))


def run(**kw):
    hit = gt = fp = det = clean = cfp = 0
    notes = collections.Counter()
    for b, ref, im, bx in cases:
        out, note, ok = reference.compare(ref, im, **kw)
        notes[note.split(" (")[0]] += 1
        if not bx:
            clean += 1
            cfp += bool(out)
            continue
        gt += len(bx)
        det += len(out)

        def touch(o, g):
            ix = max(0, min(o[2], g[3]) - max(o[0], g[1]))
            iy = max(0, min(o[3], g[4]) - max(o[1], g[2]))
            return ix * iy > 0.2 * min((o[2] - o[0]) * (o[3] - o[1]), (g[3] - g[1]) * (g[4] - g[2]))

        hit += sum(any(touch(o, g) for o in out) for g in bx)
        fp += sum(not any(touch(o, g) for g in bx) for o in out)
    return f"found {hit}/{gt}={hit / max(1, gt):.2f}  wrong boxes {fp} ({fp / max(1, len(cases) - clean):.2f}/image)  clean boards flagged {cfp}/{clean}  {dict(notes)}"


print(len(cases), "cases")
if a.grid:
    for th, tol, ma in itertools.product((30, 38, 46), (5, 7, 9), (30, 45)):
        print(th, tol, ma, run(thresh=th, tol=tol, min_area=ma))
else:
    print(run())
