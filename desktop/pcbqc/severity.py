import math

ELECTRICAL = {"short", "open_circuit"}
MOUNTING = {"missing_hole", "missing_pad"}
LARGE_PCT, LARGE_ANOMALY_PCT, REPEAT, SURE = 0.20, 0.10, 3, 0.55
ORDER = {"minor": 1, "major": 2, "critical": 3}


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u else 0


def zone(b, w, h):
    col = ("left", "centre", "right")[min(2, int((b[0] + b[2] / 2) / w * 3))]
    row = ("top", "middle", "bottom")[min(2, int((b[1] + b[3] / 2) / h * 3))]
    return "centre" if (row, col) == ("middle", "centre") else f"{row} {col}"


def _near(f, o):
    a, b = f["bbox"], o["bbox"]
    d = math.hypot(b[0] + b[2] / 2 - a[0] - a[2] / 2, b[1] + b[3] / 2 - a[1] - a[3] / 2)
    return o is not f and _iou(a, b) < 0.3 and d < 1.5 * max(a[2], a[3], b[2], b[3])


def grade(found, w, h):
    counts = {}
    for f in found:
        counts[f["cls"]] = counts.get(f["cls"], 0) + 1
    out = []
    for f in found:
        b = f["bbox"]
        pct = 100.0 * b[2] * b[3] / (w * h)
        if f["cls"] in ELECTRICAL:
            sev, why = "critical", "breaks the circuit electrically"
        elif f["cls"] in MOUNTING:
            sev, why = "major", "a part cannot be mounted or soldered here"
        elif f["cls"] == "anomaly":
            large = pct > LARGE_ANOMALY_PCT
            sev, why = ("major", "large difference from the reference board") if large else ("minor", "small difference from the reference board")
        elif pct > LARGE_PCT:
            sev, why = "major", "large area"
        elif counts[f["cls"]] >= REPEAT:
            sev, why = "major", f"repeats {counts[f['cls']]} times, so the process may be drifting"
        elif any(_near(f, o) for o in found):
            sev, why = "major", "close to another defect"
        else:
            sev, why = "minor", "small and isolated"
        out.append({**f, "severity": sev, "area_pct": round(pct, 3), "zone": zone(b, w, h), "reason": why})
    return sorted(out, key=lambda d: (-ORDER[d["severity"]], -d["conf"]))


def verdict(graded, problems=()):
    if not graded:
        return "RETAKE" if problems else "PASS"
    worst = max(ORDER[d["severity"]] for d in graded)
    sure = max(d["conf"] for d in graded if ORDER[d["severity"]] == worst) >= SURE
    if worst == 1 or not sure:
        return "REVIEW"
    return "REJECT" if worst == 3 else "REWORK"
