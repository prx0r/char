"""Cut a generated character sheet (flat magenta bg) into rig layers.

Input: characters/<slug>/rig.json  (art paths, cut geometry, feature anchors)
Output: characters/<slug>/layers/{body,head,jaw?,hand?}.png  (all canvas-aligned, RGBA)
"""
import json, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def key_magenta(img):
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    bg = np.median(np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3)]), axis=0)
    d = np.sqrt(((a - bg) ** 2).sum(-1))
    alpha = np.clip((d - 40) / 60.0, 0, 1)
    # despill: magenta = R and B high vs G
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    spill = np.clip(np.minimum(r, b) - g, 0, None)
    edge = (alpha < 0.999)
    k = np.where(edge, spill * (1 - alpha) * 0.9, 0)
    out = a.copy()
    out[..., 0] -= k; out[..., 2] -= k
    # second pass: kill the magenta rim light baked into the outline band (within ~8 px of the cut)
    from scipy.ndimage import binary_dilation
    band = binary_dilation(alpha < 0.5, iterations=12) & (alpha > 0)
    r2, g2, b2 = out[..., 0], out[..., 1], out[..., 2]
    sp2 = np.clip(np.minimum(r2, b2) - g2, 0, None)
    hue_mag = (r2 > g2 + 4) & (b2 > g2 + 4)
    kk = np.where(band & hue_mag, sp2 * 0.95, 0)
    out[..., 0] -= kk; out[..., 2] -= kk
    rgba = np.dstack([np.clip(out, 0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")

def poly_mask(size, polys, dilate=0):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon([tuple(x) for x in p], fill=255)
    if dilate:
        m = m.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    return m

def main(slug):
    root = os.path.join(os.path.dirname(__file__), "..", "characters", slug)
    rig = json.load(open(os.path.join(root, "rig.json")))
    art = Image.open(os.path.join(root, rig["art"]["base"]))
    W, H = art.size
    keyed = key_magenta(art)
    cut = rig["cut"]
    ycut = cut["y_cut"]
    top = [[0, 0], [W, 0], [W, ycut], [0, ycut]]
    head_m = poly_mask((W, H), [top, cut["neck_poly"]])
    # head layer: keyed pixels in head mask + hidden neck extension filled with neck colour
    head = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    head.paste(keyed, (0, 0), head_m)
    ext = poly_mask((W, H), [cut["neck_poly"]], dilate=14)
    ext_only = Image.fromarray(((np.asarray(ext) > 0) & (np.asarray(head_m) == 0)).astype(np.uint8) * 255)
    ka = np.asarray(keyed)[..., 3] > 128   # only where the body is opaque (hidden under the collar)
    ext_only = Image.fromarray((np.asarray(ext_only) * (np.arange(H)[:, None] > ycut - 20) * ka).astype(np.uint8))
    fill = Image.new("RGBA", (W, H), tuple(cut["neck_fill"]) + (255,))
    under = Image.new("RGBA", (W, H), (0, 0, 0, 0)); under.paste(fill, (0, 0), ext_only)
    head = Image.alpha_composite(under, head)
    body = keyed.copy()
    ba = np.asarray(body).copy(); ba[..., 3] = np.where(np.asarray(head_m) > 0, 0, ba[..., 3])
    body = Image.fromarray(ba, "RGBA")
    L = os.path.join(root, "layers"); os.makedirs(L, exist_ok=True)
    head.save(os.path.join(L, "head.png")); body.save(os.path.join(L, "body.png"))
    out = {"head": "layers/head.png", "body": "layers/body.png", "size": [W, H]}
    mouth = rig.get("mouth", {})
    if mouth.get("type") == "jaw":
        # jaw = head pixels between mouth line (top edge, inclusive of line) and jaw bottom, feathered bottom
        cx, cy, hw, curv = mouth["cx"], mouth["cy"], mouth["hw"], mouth["curve"]
        lt = mouth.get("line_half", 3)
        yb = mouth["jaw_bottom"]
        m = np.zeros((H, W), np.float32)
        for x in range(int(cx - hw - 30), int(cx + hw + 30)):
            u = (x - cx) / hw
            y0 = cy + curv * u * u - lt
            if abs(u) > 1: y0 = cy + curv - lt + (abs(u) - 1) * hw * 0.6  # rise toward cheeks
            for y in range(int(y0), yb + 1):
                f = 1.0 if y < yb - 18 else (yb - y) / 18.0
                m[y, x] = max(m[y, x], f)
        hm = np.asarray(head).copy().astype(np.float32)
        hm[..., 3] *= m
        Image.fromarray(hm.astype(np.uint8), "RGBA").save(os.path.join(L, "jaw.png"))
        out["jaw"] = "layers/jaw.png"
    for k, p in rig["art"].get("props", {}).items():
        im = key_magenta(Image.open(os.path.join(root, p)))
        bb = im.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
        im = im.crop(bb); im.save(os.path.join(L, k + ".png"))
        out[k] = "layers/%s.png" % k; out[k + "_size"] = list(im.size)
    json.dump(out, open(os.path.join(L, "layers.json"), "w"), indent=1)
    print(json.dumps(out))

if __name__ == "__main__":
    main(sys.argv[1])
