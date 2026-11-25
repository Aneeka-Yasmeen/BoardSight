import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

REPO = "KeenForgeAI/PKU-Market-PCB-corrected"
REVISION = "db5480263c692b04c682548fa3af6ce46ba0b626"
UA = {"User-Agent": "Mozilla/5.0"}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = os.path.realpath(a.out)
url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
meta = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))
keep = [s for s in meta["siblings"] if s["rfilename"].split("/")[0] in ("images", "labels", "ImageSets", "LICENSE", "README.md", "classes.txt")]


def get(s):
    p = os.path.join(base, *s["rfilename"].split("/"))
    if ".." in s["rfilename"].split("/"):
        return s["rfilename"]
    if os.path.exists(p) and os.path.getsize(p) == s["size"]:
        return None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for k in range(6):
        try:
            b = urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{s['rfilename']}", headers=UA), timeout=60).read()
            if len(b) == s["size"]:
                with open(p, "wb") as f:
                    f.write(b)
                return None
        except OSError:
            time.sleep(3 * (k + 1))
    return s["rfilename"]


with cf.ThreadPoolExecutor(6) as ex:
    failed = [x for x in ex.map(get, keep) if x]
print(len(keep), "files, failed:", failed)
