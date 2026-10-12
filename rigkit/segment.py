"""Part-sheet segmentation: chroma-key a flat-background sheet and split it into parts.

Every sheet is generated on a flat magenta (#FF00FF-ish) background. We
  1. estimate the background colour from the border,
  2. build a soft alpha from colour distance (with an un-mix step so the
     anti-aliased fringe does not keep a magenta tint),
  3. label 8-connected components, drop specks, and return RGBA crops.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def key_sheet(path, t0=48.0, t1=100.0):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    border = np.concatenate([im[:4].reshape(-1, 3), im[-4:].reshape(-1, 3), im[:, :4].reshape(-1, 3), im[:, -4:].reshape(-1, 3)])
    bg = np.median(border, 0)
    d = np.sqrt(((im - bg) ** 2).sum(-1))
    # magenta-ness: background is high R & B, low G. Use chroma distance plus a magenta score so
    # dark outlines (far from bg) stay opaque and pink-ish fringe goes transparent.
    alpha = np.clip((d - t0) / (t1 - t0), 0, 1)
    # un-mix fringe colour: obs = a*fg + (1-a)*bg  =>  fg = (obs - (1-a)*bg)/a
    a3 = alpha[..., None]
    fg = np.where(a3 > 0.02, (im - (1 - a3) * bg) / np.maximum(a3, 0.02), im)
    fg = np.clip(fg, 0, 255)
    # residual despill: no foreground pixel should be more magenta than its neighbours' green allows
    r, g, b = fg[..., 0], fg[..., 1], fg[..., 2]
    spill = np.minimum(r, b) - g
    edge = alpha < 0.98
    lim = np.where(edge & (spill > 30), g + 30, np.minimum(r, b))
    fg[..., 0] = np.where(edge & (spill > 30), np.minimum(r, lim), r)
    fg[..., 2] = np.where(edge & (spill > 30), np.minimum(b, lim), b)
    # house style has dark outlines: any semi-transparent edge pixel is outline, so give it the ink colour
    L0 = fg[..., 0] * 0.299 + fg[..., 1] * 0.587 + fg[..., 2] * 0.114
    inkpx = fg[(alpha > 0.98) & (L0 < 70)]
    if len(inkpx) > 50:
        ink = np.median(inkpx, 0)
        edge = (alpha > 0.0) & (alpha < 0.85)
        fg[edge] = ink
    raw = fg.astype(np.uint8)
    fg = demagenta(fg, alpha)
    return fg.astype(np.uint8), alpha, bg, raw


def demagenta(fg, alpha, dr=26, db=10):
    """The image model paints a pink rim-light (background bounce) inside outlines. Replace strongly
    magenta-hued opaque pixels with the nearest non-magenta foreground colour."""
    r, g, b = fg[..., 0], fg[..., 1], fg[..., 2]
    L = r * 0.299 + g * 0.587 + b * 0.114
    L = r * 0.299 + g * 0.587 + b * 0.114
    bad = (r - g > dr) & (b - g > db) & (alpha > 0.05)
    # near the silhouette the bounce light is weaker/desaturated: looser test inside an 9 px band
    band = ndi.distance_transform_edt(alpha > 0.5) < 18
    bad |= band & (r - g > 16) & (b - g > -8) & (alpha > 0.05) & (L >= 95)
    out = fg.copy()
    for sel, good in ((bad & (L >= 95), (~bad) & (alpha > 0.9) & (L >= 95)),     # pink rim -> interior colour
                      (bad & (L < 95), (~bad) & (alpha > 0.9) & (L < 95))):      # magenta-tinted ink -> ink
        if not sel.any() or not good.any(): continue
        _, (iy, ix) = ndi.distance_transform_edt(~good, return_indices=True)
        out[sel] = fg[iy[sel], ix[sel]]
    return out


def components(path, min_area=1500, pad=6):
    fg, alpha, bg, raw = key_sheet(path)
    solid = alpha > 0.35
    solid = ndi.binary_closing(solid, iterations=2)
    lab, n = ndi.label(solid, structure=np.ones((3, 3)))
    out = []
    H, W = alpha.shape
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if area < min_area:
            continue
        y0, y1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
        # alpha restricted to this component (dilated so soft edges survive)
        mm = np.zeros((H, W), bool)
        mm[sl] = m
        mm = ndi.binary_dilation(mm, iterations=3)
        a = (alpha * mm)[y0:y1, x0:x1]
        rgba = np.dstack([fg[y0:y1, x0:x1], (a * 255).astype(np.uint8)])
        rgb_raw = raw[y0:y1, x0:x1]
        ys, xs = np.nonzero(m)
        out.append({"bbox": [x0, y0, x1, y1], "area": area, "rgba": rgba, "rgb_raw": rgb_raw,
                    "centroid": [float(xs.mean() + sl[1].start), float(ys.mean() + sl[0].start)],
                    "w": x1 - x0, "h": y1 - y0})
    out.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return out, (W, H)


def save_rgba(rgba, path):
    Image.fromarray(rgba, "RGBA").save(path)
