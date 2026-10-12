"""Landmarks without hand-placed points.

Body joints come from part geometry (silhouette analysis of each segmented part); face features
come from MediaPipe Face Landmarker when the head is human enough for it to fire, else from a
cartoon-feature detector (eye blobs = bright enclosed regions, mouth = widest interior ink stroke).
"""
import os, json
import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def lum(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def width_profile(alpha, thr=0.5):
    m = alpha > thr
    rows = []
    for y in range(m.shape[0]):
        xs = np.nonzero(m[y])[0]
        rows.append((xs.min(), xs.max()) if len(xs) else None)
    return rows


def bottom_center(alpha, frac=0.06):
    m = alpha > 0.5
    ys, xs = np.nonzero(m)
    y1 = ys.max(); band = ys > y1 - max(4, frac * (ys.max() - ys.min()))
    return float(xs[band].mean()), float(y1)


def top_center(alpha, frac=0.06):
    m = alpha > 0.5
    ys, xs = np.nonzero(m)
    y0 = ys.min(); band = ys < y0 + max(4, frac * (ys.max() - ys.min()))
    return float(xs[band].mean()), float(y0)


def chin_row(alpha):
    """Row where the head narrows into the neck (first row below the widest row whose width is
    within 12% of the neck width measured near the bottom)."""
    rows = width_profile(alpha)
    ws = np.array([(r[1] - r[0]) if r else 0 for r in rows], float)
    ys = np.nonzero(ws)[0]; y0, y1 = ys.min(), ys.max()
    neck_w = np.median(ws[int(y1 - (y1 - y0) * 0.12):y1 + 1])
    wide = int(y0 + np.argmax(ws[y0:y1]))
    for y in range(wide, y1):
        if ws[y] <= neck_w * 1.12:
            return y, float(neck_w), wide
    return int(y0 + (y1 - y0) * 0.7), float(neck_w), wide


def edge_distance(alpha):
    return ndi.distance_transform_edt(alpha > 0.5)


def eye_blobs(rgb, alpha, max_eyes=2):
    """Bright, enclosed, compact blobs in the upper 65% of the head = painted eyes.
    Returns list of dicts with mask (bool, full-size), centroid, radii, colours."""
    H, W = alpha.shape
    L = lum(rgb.astype(float))
    ys, xs = np.nonzero(alpha > 0.5); top, bot = ys.min(), ys.max()
    region = np.zeros_like(alpha, bool); region[top:int(top + (bot - top) * 0.62)] = True
    ed = edge_distance(alpha)
    bright = (L > 208) & (alpha > 0.9) & region
    bright = ndi.binary_opening(bright, iterations=1)
    lab, n = ndi.label(bright)
    cands = []
    for i in range(1, n + 1):
        m = lab == i
        a = m.sum()
        if a < 250: continue
        m2 = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))
        yy, xx = np.nonzero(m2)
        h, w = np.ptp(yy) + 1, np.ptp(xx) + 1
        fill = m2.sum() / float(h * w)
        if fill < 0.45 or max(h, w) / max(1, min(h, w)) > 3.2: continue
        cands.append((m2.sum(), m2))
    cands.sort(key=lambda c: -c[0])
    eyes = []
    for a, m in cands[:max_eyes]:
        if eyes and a < 0.12 * eyes[0]["area"]: break
        yy, xx = np.nonzero(m)
        px = rgb[m & (L > 150)]
        eyes.append({"area": int(a), "mask": m, "cx": float(xx.mean()), "cy": float(yy.mean()),
                     "rx": float(np.ptp(xx) / 2 + 0.5), "ry": float(np.ptp(yy) / 2 + 0.5),
                     "bbox": [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                     "color": [int(v) for v in np.median(px, 0)] if len(px) else [240, 230, 170]})
    eyes.sort(key=lambda e: e["cx"])
    return eyes


def mouth_line(rgb, alpha, eyes, chin_y):
    """Widest dark interior stroke between the eyes and the chin -> quadratic y(x)."""
    L = lum(rgb.astype(float))
    ed = edge_distance(alpha)
    ink = (L < 95) & (alpha > 0.8)
    ys, xs = np.nonzero(alpha > 0.5); top = ys.min()
    ey = max(e["cy"] + e["ry"] for e in eyes) if eyes else top + (chin_y - top) * 0.45
    band = np.zeros_like(ink); band[int(ey + 6):int(chin_y)] = True
    cand = ink & band & (ed > 7)
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    best = None
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        span = np.ptp(xx)
        if span < 40 or np.ptp(yy) > max(30, span * 0.45): continue
        if best is None or span > best[0]: best = (span, xx, yy)
    if best is None: return None
    _, xx, yy = best
    # centreline: mean y per x
    ux = np.unique(xx); cy = np.array([yy[xx == u].mean() for u in ux])
    c = np.polyfit(ux, cy, 2)
    x0, x1 = float(ux.min()), float(ux.max())
    # extend to the silhouette along the line (the trimmed band near the outline)
    def y_at(x): return float(np.polyval(c, x))
    H, W = alpha.shape
    for direction in (-1, 1):
        x = x0 if direction < 0 else x1
        while 0 < x < W - 1 and ed[int(min(H - 1, max(0, y_at(x)))), int(x)] > 2 and abs(x - (x0 + x1) / 2) < W:
            x += direction
        if direction < 0: x0 = float(x + 3)
        else: x1 = float(x - 3)
    return {"coef": [float(v) for v in c], "x0": x0, "x1": x1, "cy": y_at((x0 + x1) / 2)}


def skin_color(rgb, alpha, eyes):
    L = lum(rgb.astype(float))
    m = (alpha > 0.95) & (L > 100) & (L < 215)
    for e in eyes: m &= ~ndi.binary_dilation(e["mask"], iterations=6)
    ys, xs = np.nonzero(alpha > 0.5)
    band = np.zeros_like(m); band[ys.min():int(ys.min() + 0.55 * np.ptp(ys))] = True
    px = rgb[m & band]
    return [int(v) for v in np.median(px, 0)] if len(px) else [180, 140, 110]


def ink_color(rgb, alpha):
    L = lum(rgb.astype(float)); px = rgb[(L < 70) & (alpha > 0.9)]
    return [int(v) for v in np.median(px, 0)] if len(px) else [58, 29, 18]


# ---------------- MediaPipe (optional, human faces) ----------------
_FL = None
import atexit
def _close():
    global _FL
    try:
        if _FL is not None: _FL.close()
    except Exception: pass
    _FL = None
atexit.register(_close)
def mediapipe_face(rgb):
    """Return dict with eye contours + lips from MediaPipe Face Landmarker, or None."""
    global _FL
    model = os.path.join(ROOT, ".cache", "mediapipe", "face_landmarker.task")
    if not os.path.exists(model): return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        if _FL is None:
            _FL = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), num_faces=1))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb[..., :3]))
        r = _FL.detect(img)
        if not r.face_landmarks: return None
        H, W = rgb.shape[:2]
        P = np.array([[p.x * W, p.y * H] for p in r.face_landmarks[0]])
        return P
    except Exception as e:
        print("mediapipe face failed:", e); return None

# MediaPipe face-mesh index sets
MP_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]   # subject's right eye (image left)
MP_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MP_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
