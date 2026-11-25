import argparse
import os

import cv2

from pcbqc import logdb, pipeline


def show(r):
    source = "Llama" if r["llama"] else "playbook"
    print(f"\nVERDICT: {r['verdict']} | {len(r['defects'])} defect(s) | vision {r['vision_ms']} ms | advice {r['advice_ms']} ms ({source}) | {r['reference']}")
    for d in r["defects"]:
        print(f"  - {d['cls']:16s} {d['severity']:8s} {d['zone']:14s} conf={d['conf']:.2f} {d['reason']}")
    print(r["advice"] + f"\nSaved: {r['image']}")


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
i = sp.add_parser("image")
i.add_argument("path")
i.add_argument("--golden")
c = sp.add_parser("camera")
c.add_argument("--golden")
c.add_argument("--cam", type=int, default=0)
sp.add_parser("report")
a = ap.parse_args()
gold = cv2.imread(a.golden) if getattr(a, "golden", None) else None
if a.cmd == "image":
    img = cv2.imread(a.path)
    if img is None:
        raise SystemExit("Could not open " + a.path)
    show(pipeline.inspect(img, gold, os.path.splitext(os.path.basename(a.path))[0]))
elif a.cmd == "camera":
    cap, n = cv2.VideoCapture(a.cam), 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imshow("PCB inspection (space = inspect, q = quit)", f)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            n += 1
            show(pipeline.inspect(f, gold, f"cam{n:04d}"))
        if k == ord("q"):
            break
else:
    for v, n, ms in logdb.summary("inspections.db"):
        print(f"{v:8s} count={n:4d} avg_vision_ms={ms}")
