import argparse
import itertools
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pcbdefect", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.load(open(os.path.join(a.pcbdefect, "annotation", "_annotations.coco.json")))
orb = cv2.ORB_create(1200)
feats = {}
for i in d["images"]:
    g = cv2.imread(os.path.join(a.pcbdefect, "images", i["file_name"]), 0)
    s = 700 / max(g.shape)
    g = cv2.GaussianBlur(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (7, 7), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k, e = orb.detectAndCompute(b, None)
    feats[i["extra"]["name"][:-4]] = (np.float32([p.pt for p in k]), e)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
pairs = []
for p, q in itertools.combinations(sorted(feats), 2):
    (pa, ea), (pb, eb) = feats[p], feats[q]
    good = [m[0] for m in matcher.knnMatch(ea, eb, 2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    n = 0
    if len(good) >= 12:
        _, mask = cv2.estimateAffinePartial2D(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]], method=cv2.RANSAC, ransacReprojThreshold=6)
        n = int(mask.sum()) if mask is not None else 0
    pairs.append((n, p, q))
json.dump(sorted(pairs, reverse=True), open(a.out, "w"))
print(len(pairs), "pairs,", sum(n >= 60 for n, _, _ in pairs), "treated as the same design")
