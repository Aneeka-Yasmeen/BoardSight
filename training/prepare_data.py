import argparse
import collections
import glob
import json
import os
import re
import shutil

import cv2
import numpy as np

from common import CID, CLASSES, colourise, phone_sim, to_yolo

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
T = 640
counts = collections.Counter()
boxes_n = collections.defaultdict(collections.Counter)
tests = []

for s in ("train", "val"):
    for k in ("images", "labels"):
        shutil.rmtree(os.path.join(a.out, k, s), ignore_errors=True)
        os.makedirs(os.path.join(a.out, k, s))


def save(split, name, img, boxes):
    h, w = img.shape[:2]
    lab = to_yolo(boxes, w, h)
    cv2.imwrite(os.path.join(a.out, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
        f.write(lab)
    counts[(split, name.split("_")[0])] += 1
    for row in lab.split("\n"):
        if row:
            boxes_n[split][CLASSES[int(row.split()[0])]] += 1


def crop(img, boxes, long_side, centre=None):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    sz = T / s
    best = None
    for _ in range(12):
        if centre is not None:
            cx = centre[0] + rng.uniform(-0.38, 0.38) * sz
            cy = centre[1] + rng.uniform(-0.38, 0.38) * sz
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        x0 = int(np.clip(cx - sz / 2, 0, max(0, w - sz)))
        y0 = int(np.clip(cy - sz / 2, 0, max(0, h - sz)))
        keep, partial = [], 0
        for c, x1, y1, x2, y2 in boxes:
            vx = max(0, min(x2, x0 + sz) - max(x1, x0))
            vy = max(0, min(y2, y0 + sz) - max(y1, y0))
            r = vx * vy / max(1e-6, (x2 - x1) * (y2 - y1))
            if r >= 0.55:
                keep.append((c, (max(x1, x0) - x0) * s, (max(y1, y0) - y0) * s, (min(x2, x0 + sz) - x0) * s, (min(y2, y0 + sz) - y0) * s))
            elif r > 0.08:
                partial += 1
        if best is None or partial < best[0]:
            best = (partial, x0, y0, keep)
        if partial == 0:
            break
    partial, x0, y0, keep = best
    if partial and centre is None:
        return None
    c = img[y0:int(y0 + sz), x0:int(x0 + sz)]
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    can = np.full((T, T, 3), 114, np.uint8)
    can[: min(T, c.shape[0]), : min(T, c.shape[1])] = c[:T, :T]
    return can, keep


def many(split, stem, img, boxes, per_box, clean, sim_p):
    n = 0
    jobs = [((b[1] + b[3]) / 2, (b[2] + b[4]) / 2) for b in boxes for _ in range(per_box)] + [None] * clean
    for ctr in jobs:
        r = crop(img, boxes, int(rng.choice([1500, 1900, 2300, 2700, 3200])), ctr)
        if r is None or (ctr is None and r[1]):
            continue
        im, bx = r
        if rng.random() < sim_p:
            im, bx = phone_sim(im, bx, rng, rng.uniform(0.5, 1.2))
        save(split, f"{stem}_{n:03d}", im, bx)
        n += 1


R = os.path.join(a.data, "PCB_Defect", "PCB_Defect")
d = json.load(open(os.path.join(R, "annotation", "_annotations.coco.json")))
par = {i["extra"]["name"][:-4]: i["extra"]["name"][:-4] for i in d["images"]}


def find(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


for n, p, q in json.load(open(a.pairs)):
    if n >= 60:
        par[find(p)] = find(q)
cl = collections.defaultdict(list)
for i in d["images"]:
    cl[find(i["extra"]["name"][:-4])].append(i)
order = sorted(cl)
rng.shuffle(order)
assign, n_test, n_val = {}, 0, 0
for k in order:
    if len(cl[k]) <= 4 and n_test < 50:
        assign[k] = "test"
        n_test += len(cl[k])
    elif len(cl[k]) <= 4 and n_val < 22:
        assign[k] = "val"
        n_val += len(cl[k])
    else:
        assign[k] = "train"
ann = collections.defaultdict(list)
names = {c["id"]: c["name"] for c in d["categories"]}
for x in d["annotations"]:
    bx, by, bw, bh = x["bbox"]
    ann[x["image_id"]].append((CID[names[x["category_id"]]], bx, by, bx + bw, by + bh))
for k in order:
    for i in cl[k]:
        p = os.path.join(R, "images", i["file_name"])
        if assign[k] == "test":
            tests.append({"set": "pcbdefect", "image": os.path.abspath(p), "mode": "tile", "boxes": ann[i["id"]]})
            continue
        img = cv2.imread(p)
        if img.shape[:2] != (i["height"], i["width"]):
            raise SystemExit("size mismatch " + p)
        tr = assign[k] == "train"
        many(assign[k], "pcbdefect_" + i["file_name"][11:-4], img, ann[i["id"]], 3 if tr else 1, 4 if tr else 2, 0.6 if tr else 0.5)
print("pcbdefect clusters", len(cl), dict(collections.Counter(assign.values())), "test images", n_test, "val", n_val)

P = os.path.join(a.data, "pku")
pcl = [x.strip() for x in open(os.path.join(P, "classes.txt")) if x.strip()]
for p in sorted(glob.glob(os.path.join(P, "images", "*.jpg"))):
    stem = os.path.basename(p)[:-4]
    board = stem[:2]
    lp = os.path.join(P, "labels", stem + ".txt")
    img = cv2.imread(p) if os.path.exists(lp) else None
    if img is None:
        continue
    h, w = img.shape[:2]
    bx = []
    for row in open(lp):
        c, x, y, bw, bh = map(float, row.split()[:5])
        bx.append((CID[pcl[int(c)]], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    if board in ("10", "11", "12"):
        tests.append({"set": "pku", "image": os.path.abspath(p), "mode": "tile", "boxes": bx})
    else:
        v = board == "09"
        many("val" if v else "train", "pku_" + stem, img, bx, 1 if v else 2, 1 if v else 2, 0.5 if v else 0.6)

D = os.path.join(a.data, "DeepPCB", "DeepPCB-master", "PCBData")


def dp(list_name):
    for line in open(os.path.join(D, list_name)):
        i, lab = line.split()
        yield os.path.join(D, i[:-4] + "_test.jpg"), os.path.join(D, i[:-4] + "_temp.jpg"), os.path.join(D, lab), i.split("/")[0]


def dp_boxes(lp):
    out = []
    for row in open(lp):
        v = re.split(r"[ ,]+", row.strip())
        if len(v) >= 5:
            x1, y1, x2, y2, t = map(int, v[:5])
            if 1 <= t <= 6:
                out.append((t - 1, x1, y1, x2, y2))
    return out


tv, te = list(dp("trainval.txt")), list(dp("test.txt"))
ov = {x[3] for x in tv} & {x[3] for x in te}
print("deeppcb trainval", len(tv), "test", len(te), "groups in test", len({x[3] for x in te}), "shared with trainval", len(ov))
vg = set(sorted({x[3] for x in tv})[-1:])
for k, (t, tmp, lab, g) in enumerate(tv):
    sp = "val" if g in vg else "train"
    im, bx = cv2.imread(t), dp_boxes(lab)
    stem = "deeppcb_" + os.path.basename(t)[:-9]
    save(sp, stem, im, bx)
    if sp == "train":
        ci, cb = phone_sim(colourise(im[:, :, 0], rng), bx, rng, rng.uniform(0.4, 1.0))
        save(sp, stem + "c", ci, cb)
        if k % 2 == 0:
            z = cv2.imread(tmp)
            z = z if rng.random() < 0.4 else phone_sim(colourise(z[:, :, 0], rng), [], rng, 0.8)[0]
            save(sp, stem + "n", z, [])
    elif k % 2 == 0:
        save(sp, stem + "n", cv2.imread(tmp), [])
for t, tmp, lab, g in te:
    tests.append({"set": "deeppcb", "image": os.path.abspath(t), "mode": "full", "boxes": dp_boxes(lab)})
    tests.append({"set": "deeppcb_clean", "image": os.path.abspath(tmp), "mode": "full", "boxes": []})

S = os.path.join(a.data, "DsPCBSDplus", "Data_YOLO")
DM = {0: "short", 1: "spur", 2: "spurious_copper", 3: "open_circuit", 4: "mouse_bite"}


def ds(split):
    for p in sorted(glob.glob(os.path.join(S, "images", split, "*.jpg"))):
        with open(os.path.join(S, "labels", split, os.path.basename(p)[:-4] + ".txt")) as f:
            rows = [list(map(float, r.split()[:5])) for r in f if r.strip()]
        if rows and all(int(r[0]) in DM for r in rows):
            yield p, [(CID[DM[int(r[0])]], r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2) for r in rows]


items = list(ds("train"))
cut = int(len(items) * 0.95)


def quad(split, part):
    idx = rng.permutation(len(part))
    for q in range(0, len(idx) - 3, 4):
        can, bx = np.zeros((T, T, 3), np.uint8), []
        for j in range(4):
            p, b = part[idx[q + j]]
            ox, oy = (j % 2) * 320, (j // 2) * 320
            can[oy:oy + 320, ox:ox + 320] = cv2.resize(cv2.imread(p), (320, 320), interpolation=cv2.INTER_CUBIC)
            bx += [(c, ox + x1 * 320, oy + y1 * 320, ox + x2 * 320, oy + y2 * 320) for c, x1, y1, x2, y2 in b]
        if split == "train" and rng.random() < 0.5:
            can, bx = phone_sim(can, bx, rng, 0.8)
        save(split, f"dspcb_{split}{q:05d}", can, bx)


quad("train", items[:cut])
quad("val", items[cut:])
for p, b in ds("val"):
    h, w = cv2.imread(p).shape[:2]
    tests.append({"set": "dspcb", "image": os.path.abspath(p), "mode": "full", "boxes": [(c, x1 * w, y1 * h, x2 * w, y2 * h) for c, x1, y1, x2, y2 in b]})

json.dump(tests, open(os.path.join(a.out, "tests.json"), "w"))
with open(os.path.join(a.out, "pcb.yaml"), "w") as f:
    f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
print(dict(sorted(counts.items())))
for s in boxes_n:
    print(s, dict(boxes_n[s]))
print(collections.Counter(t["set"] for t in tests))
