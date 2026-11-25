import cv2
import numpy as np

SIDE, MIN_INLIERS, THRESH, TOL, MIN_AREA, MAX_FINDINGS = 2400, 40, 26, 5, 40, 25


def gray(b):
    s = min(1.0, SIDE / max(b.shape[:2]))
    g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return (cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g), s


def flat(g):
    size, small = (g.shape[1], g.shape[0]), (g.shape[1] // 8, g.shape[0] // 8)
    f = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    mean = cv2.GaussianBlur(cv2.resize(f, small, interpolation=cv2.INTER_AREA), (0, 0), 5.0)
    d = f - cv2.resize(mean, size, interpolation=cv2.INTER_LINEAR)
    v = cv2.GaussianBlur(cv2.resize(d * d, small, interpolation=cv2.INTER_AREA), (0, 0), 7.5)
    v = cv2.resize(np.maximum(np.sqrt(v), 12), size, interpolation=cv2.INTER_LINEAR)
    return np.clip(np.round(d / v * 45 + 128), 0, 255).astype(np.uint8)


def compare(ref, test, thresh=THRESH, tol=TOL, min_area=MIN_AREA):
    g0, _ = gray(ref)
    t0, ts = gray(test)
    g, t = flat(g0), flat(t0)
    orb = cv2.ORB_create(4000)
    kg, dg = orb.detectAndCompute(g, None)
    kt, dt = orb.detectAndCompute(t, None)
    if dg is None or dt is None:
        return [], "no detail", False
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dg, dt, 2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < MIN_INLIERS:
        return [], "not matched", False
    hm, inl = cv2.findHomography(np.float32([kg[m.queryIdx].pt for m in good]), np.float32([kt[m.trainIdx].pt for m in good]), cv2.RANSAC, 3.0, maxIters=3000, confidence=0.995)
    n = 0 if hm is None else int(inl.sum())
    if n < MIN_INLIERS:
        return [], f"not matched ({n})", False
    size = (t0.shape[1], t0.shape[0])
    warped = cv2.warpPerspective(g0, hm, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(np.ones_like(g0), hm, size, flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((25, 25), np.uint8))
    on = np.int32([kt[m.trainIdx].pt for m, f in zip(good, inl.ravel()) if f])
    area = np.zeros_like(t0)
    cv2.fillConvexPoly(area, cv2.convexHull(on), 1)
    mask = cv2.bitwise_and(mask, cv2.dilate(area, np.ones((17, 17), np.uint8)))
    w = flat(warped)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol, tol))
    d = cv2.max(cv2.subtract(t, cv2.dilate(w, k)), cv2.subtract(cv2.erode(w, k), t))
    b = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)[1]
    b[mask == 0] = 0
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    out = []
    for c in cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= min_area:
            x, y, ww, hh = cv2.boundingRect(c)
            out.append(((x - 6) / ts, (y - 6) / ts, (x + ww + 6) / ts, (y + hh + 6) / ts))
    if len(out) > MAX_FINDINGS:
        return [], f"ignored ({len(out)} differences)", False
    return out, f"matched ({n})", True
