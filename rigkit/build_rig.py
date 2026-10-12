#!/usr/bin/env python3
"""Part sheets -> rig.json (schema rigkit/rig.schema.json, version oddrig/1).

  python3 rigkit/build_rig.py <slug>

Reads characters/<slug>/onboard.json:
  sheets.heads  : 3 head views stacked vertically (front, 3/4 toward screen-left, profile toward screen-left)
  sheets.parts  : head + torso (no arms) + 2 straight arms + 2 rest hands + hips/trousers, exploded
  sheets.hands  : 6 right-hand poses in a 2x3 grid (layout in onboard.json)
Writes characters/<slug>/rig/{rig.json, tex/*.png, validate.png}.
No coordinate in the output is hand-placed: joints come from part silhouettes, face features from
the cartoon-feature detector (MediaPipe is tried first and recorded when it fires).
"""
import os, sys, json, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import segment, features as F

SCHEMA_VERSION = "oddrig/1"


def smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


def resize_rgba(rgba, s):
    if abs(s - 1) < 1e-3: return rgba
    im = Image.fromarray(rgba, "RGBA")
    # premultiply to avoid dark fringes when resampling
    a = np.asarray(im).astype(np.float32); a[..., :3] *= a[..., 3:4] / 255.0
    pm = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
    W, H = im.size; pm = pm.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
    b = np.asarray(pm).astype(np.float32); al = b[..., 3:4] / 255.0
    b[..., :3] = np.where(al > 0.004, b[..., :3] / np.maximum(al, 0.004), 0)
    return np.clip(b, 0, 255).astype(np.uint8)


def grid_mesh(alpha, step, origin, weight_fn, uv_size, mirror=False):
    """Regular grid over the opaque area. Returns vertices (rig space), uvs, triangles, weights.
    origin: rig-space position of texture pixel (0,0). weight_fn(x_rig, y_rig) -> [[bone, w], ...]"""
    H, W = alpha.shape
    occ = ndi.binary_dilation(alpha > 0.02, iterations=2)
    nx, ny = int(math.ceil(W / step)) + 1, int(math.ceil(H / step)) + 1
    cell = np.zeros((ny - 1, nx - 1), bool)
    for j in range(ny - 1):
        for i in range(nx - 1):
            y0, x0 = int(j * step), int(i * step)
            if occ[y0:y0 + step + 1, x0:x0 + step + 1].any(): cell[j, i] = True
    idx = -np.ones((ny, nx), int); verts, uvs, wts, tris = [], [], [], []
    def vid(i, j):
        if idx[j, i] < 0:
            px, py = min(i * step, W), min(j * step, H)
            idx[j, i] = len(verts)
            x, y = origin[0] + px, origin[1] + py
            verts.append([round(float(x), 2), round(float(y), 2)])
            u = px / uv_size[0]; v = py / uv_size[1]
            uvs.append([round(1 - u if mirror else u, 5), round(v, 5)])
            wts.append(weight_fn(x, y))
        return idx[j, i]
    for j in range(ny - 1):
        for i in range(nx - 1):
            if not cell[j, i]: continue
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris += [int(a), int(b), int(c), int(a), int(c), int(d)]
    return {"vertices": verts, "uvs": uvs, "triangles": tris, "weights": wts}


def norm_w(ws):
    ws = [[b, float(w)] for b, w in ws if w > 1e-3]
    s = sum(w for _, w in ws) or 1.0
    ws = sorted(ws, key=lambda x: -x[1])[:4]
    s = sum(w for _, w in ws)
    return [[b, round(w / s, 4)] for b, w in ws]


def flood(rgb, alpha, seed, tol, limit_mask=None):
    sx, sy = int(seed[0]), int(seed[1])
    c = rgb[sy, sx].astype(float)
    d = np.sqrt(((rgb.astype(float) - c) ** 2).sum(-1))
    m = (d < tol) & (alpha > 0.5)
    if limit_mask is not None: m &= limit_mask
    lab, _ = ndi.label(m)
    if lab[sy, sx] == 0: return np.zeros_like(m)
    return lab == lab[sy, sx]


def build(slug):
    cdir = os.path.join(ROOT, "characters", slug)
    cfg = json.load(open(os.path.join(cdir, "onboard.json")))
    out = os.path.join(cdir, "rig"); tex = os.path.join(out, "tex"); os.makedirs(tex, exist_ok=True)
    S = cfg["sheets"]
    report = {"landmarks": {}}

    # ---------------- parts sheet ----------------
    comps, (SW, SH) = segment.components(os.path.join(ROOT, S["parts"]))
    torso = max(comps, key=lambda c: c["area"])
    tx0, ty0, tx1, ty1 = torso["bbox"]; tcx = (tx0 + tx1) / 2
    rest = [c for c in comps if c is not torso]
    head_ref = max([c for c in rest if c["centroid"][1] < ty0 + 40 and abs(c["centroid"][0] - tcx) < (tx1 - tx0) * 0.3],
                   key=lambda c: c["area"], default=None)
    arms = sorted([c for c in rest if c["h"] / c["w"] > 2.4 and c["h"] > 0.5 * torso["h"]], key=lambda c: c["centroid"][0])
    hips = max([c for c in rest if c["bbox"][1] > ty1 - 0.25 * torso["h"] and abs(c["centroid"][0] - tcx) < torso["w"] * 0.3], key=lambda c: c["area"], default=None)
    ref_hands = [c for c in rest if c not in arms and c is not head_ref and c is not hips and c["area"] < torso["area"] * 0.2]
    assert len(arms) == 2, "expected two straight arms on the parts sheet, got %d" % len(arms)
    report["parts_sheet"] = {"torso": torso["bbox"], "head_ref": head_ref and head_ref["bbox"], "arms": [a["bbox"] for a in arms],
                             "hips": hips and hips["bbox"], "ref_hands": [h["bbox"] for h in ref_hands]}

    # rig space == parts-sheet pixel space
    trgb, ta = torso["rgb_raw"], torso["rgba"][..., 3] / 255.0
    th, tw = ta.shape
    rows = F.width_profile(ta)
    ws = np.array([(r[1] - r[0]) if r else 0 for r in rows], float)
    top_half = ws[: th // 2]
    sh_row = int(np.argmax(top_half >= 0.9 * top_half.max()))
    shoulder_y = ty0 + sh_row
    xl, xr = tx0 + rows[sh_row][0], tx0 + rows[sh_row][1]

    # collar hole: flood from the first non-ink pixel below the top outline at the centre column
    hole = np.zeros_like(ta, bool); collar_back = np.zeros_like(ta, bool)
    col0 = int(tw / 2); ys_c = np.nonzero(ta[:, col0] > 0.5)[0]
    notch = len(ys_c) and ys_c.min() > 0.04 * th          # neck opening already transparent on the sheet
    report["landmarks"]["neck_notch"] = bool(notch)
    if cfg.get("collar_hole", True) and not notch:
        col = int(tw / 2); L = F.lum(trgb.astype(float)); y = 0
        while y < th and ta[y, col] < 0.5: y += 1
        # colour runs down the centre column; the collar opening is the first non-ink run >= 12 px
        seed = None; run0 = None; prev = None
        for yy in range(y, int(th * 0.2)):
            px = trgb[yy, col].astype(int)
            if L[yy, col] < 110: run0 = None; prev = None; continue
            if prev is None or np.abs(px - prev).sum() > 40: run0 = yy
            prev = px
            if yy - run0 >= 12: seed = (col, run0 + 6); break
        if seed is not None:
            lim = np.zeros_like(ta, bool); lim[: int(th * 0.2)] = True
            hole = flood(trgb, ta, seed, 34, lim)
            if hole.sum() < 300 or hole.sum() > 0.08 * ta.size: hole[:] = False
        if hole.any():
            hole = ndi.binary_dilation(hole, iterations=2) & (ta > 0)
            hy, hx = np.nonzero(hole)
            hcy = hy.mean()
            cb = np.zeros_like(hole); cb[: int(hcy), hx.min():hx.max() + 1] = True
            collar_back = cb & (ta > 0.02) & ~hole
    if hole.any():
        hy, hx = np.nonzero(hole)
        neck_base = (tx0 + hx.mean(), ty0 + hy.mean() + 0.25 * np.ptp(hy))
        hole_w = np.ptp(hx)
    elif notch:
        # the notch: transparent pixels inside the torso's top band, enclosed left/right by cloth
        band = ta[: int(th * 0.25)] < 0.5; rows_ = []
        for yy in range(band.shape[0]):
            xs_ = np.nonzero(ta[yy] > 0.5)[0]
            if len(xs_) < 2: continue
            inside = band[yy, xs_.min():xs_.max()]
            if inside[int(len(inside) * 0.3):int(len(inside) * 0.7)].any(): rows_.append(yy)
        ys_n = np.array(rows_) if rows_ else np.array([int(th * 0.05)])
        nb_y = ys_c.min()
        neck_base = (tx0 + col0, ty0 + nb_y - 0.1 * (nb_y - ys_n.min()))
        hole_w = 0.3 * tw
    else:
        tcx_, ttop = F.top_center(ta); neck_base = (tx0 + tcx_, ty0 + ttop + 0.06 * th); hole_w = 0.3 * tw
    report["landmarks"]["collar_hole_px"] = int(hole.sum())

    # secondary (tie): flood from below the collar on the centre line
    tie = np.zeros_like(ta, bool); tie_info = None
    for sec in cfg.get("secondary", []):
        if sec.get("name") != "tie": continue
        sy0 = int(neck_base[1] - ty0 + (0.12 * hole_w if hole.any() else 0.08 * th))
        col = int(neck_base[0] - tx0)
        best = None
        for yy in range(sy0, int(th * 0.4)):
            px = trgb[yy, col].astype(int); sat = px.max() - px.min()
            if sat > 60 and F.lum(px[None].astype(float))[0] > 90: best = (col, yy + 4); break
        if best:
            lim = np.zeros_like(ta, bool); lim[: int(th * 0.85)] = True
            tie = flood(trgb, ta, best, sec.get("tol", 46), lim)
            tie = ndi.binary_closing(tie, iterations=4)
            if tie.sum() < 500: tie[:] = False
        if tie.any():
            yy, xx = np.nonzero(tie)
            tie_info = {"top": (tx0 + xx[yy < yy.min() + 15].mean(), ty0 + yy.min()), "bottom": (tx0 + xx[yy > yy.max() - 15].mean(), ty0 + yy.max())}
    report["landmarks"]["tie_px"] = int(tie.sum())

    # ---------------- arms ----------------
    arm_info = {}
    for side, comp in (("r", arms[0]), ("l", arms[1])):        # screen-left arm is the character's right
        a = comp["rgba"][..., 3] / 255.0; x0, y0 = comp["bbox"][:2]
        aw = np.median([r[1] - r[0] for r in F.width_profile(a) if r])
        tcx_, ttop = F.top_center(a); bcx, bbot = F.bottom_center(a)
        cuff = F.width_profile(a)[int(bbot) - 6]
        arm_info[side] = {"comp": comp, "w": float(aw), "top": (x0 + tcx_, y0 + ttop), "wrist_px": (bcx, bbot), "cuff_w": float(cuff[1] - cuff[0]) if cuff else aw * 0.7}
    # assemble: arm top cap sits on the torso shoulder corner; cap centre = pivot
    for side in ("r", "l"):
        ai = arm_info[side]; aw = ai["w"]
        edge = xl if side == "r" else xr
        px = edge + (aw * 0.34 if side == "r" else -aw * 0.34)
        py = shoulder_y + aw * 0.14
        comp = ai["comp"]; x0, y0 = comp["bbox"][:2]
        cap_center_tex = (ai["top"][0] - x0, ai["top"][1] - y0 + aw * 0.5)
        ai["origin"] = (px - cap_center_tex[0], py - cap_center_tex[1])
        ai["shoulder"] = (px, py)
        ai["wrist"] = (ai["origin"][0] + ai["wrist_px"][0], ai["origin"][1] + ai["wrist_px"][1] - 4)
        ai["elbow"] = (px + (ai["wrist"][0] - px) * 0.49, py + (ai["wrist"][1] - py) * 0.49)

    # ---------------- head views ----------------
    hcomps, _ = segment.components(os.path.join(ROOT, S["heads"]))
    hcomps = sorted(hcomps, key=lambda c: c["centroid"][1])[: len(cfg["head_views"])]
    front = hcomps[0]
    ref_w = head_ref["w"] if head_ref else front["w"] * 0.9
    hs = ref_w / front["w"]
    report["head_scale"] = round(hs, 4)
    views = []
    for name, comp in zip(cfg["head_views"], hcomps):
        rgba = resize_rgba(comp["rgba"], hs)
        rgb, a = rgba[..., :3], rgba[..., 3] / 255.0
        bx, by = F.bottom_center(a)
        cy_, neck_w, wide = F.chin_row(a)
        mp = F.mediapipe_face(rgb) if cfg.get("mediapipe", True) else None
        eyes, mouth = (F.mp_features(rgb, a, mp) if mp is not None else ([], None))
        if mp is None or not eyes:
            eyes = F.eye_blobs(rgb, a)
        if mouth is None:
            mouth = F.mouth_line(rgb, a, eyes, cy_)
        views.append({"name": name, "rgba": rgba, "bottom": (bx, by), "chin_y": cy_, "neck_w": neck_w, "eyes": eyes, "mouth": mouth,
                      "skin": F.skin_color(rgb, a, eyes), "ink": F.ink_color(rgb, a), "mediapipe": mp is not None})
    # vector mouths: a view without a MediaPipe mouth gets the front mouth transferred through the eye frame
    if cfg.get("mouth", {}).get("type", "jaw") != "jaw" and views[0]["mediapipe"] and views[0]["mouth"] and len(views[0]["eyes"]) == 2:
        f0 = views[0]; fe = f0["eyes"]; fmid = ((fe[0]["cx"] + fe[1]["cx"]) / 2, (fe[0]["cy"] + fe[1]["cy"]) / 2); fd = fe[1]["cx"] - fe[0]["cx"]
        fm = f0["mouth"]; fmc = ((fm["x0"] + fm["x1"]) / 2, fm["cy"]); fw = fm["x1"] - fm["x0"]
        for v in views[1:]:
            if v["mediapipe"] or not v["eyes"]: 
                continue
            e = v["eyes"]
            if len(e) == 2:
                mid = ((e[0]["cx"] + e[1]["cx"]) / 2, (e[0]["cy"] + e[1]["cy"]) / 2); sx = (e[1]["cx"] - e[0]["cx"]) / fd
                yaw = cfg.get("view_yaw", {}).get(v["name"], 0)
                cx = mid[0] + (fmc[0] - fmid[0]) * sx + yaw * fd * 0.35; cy = mid[1] + (fmc[1] - fmid[1])
                w = fw * max(0.55, sx)
                v["mouth"] = {"coef": [0.0, 0.0, cy], "x0": cx - w / 2, "x1": cx + w / 2, "cy": cy, "fill": fm.get("fill"), "transferred": True}
            else:
                v["mouth"] = None     # profile without detection: no mouth drawn (hidden side)
    # neck depth inside the collar: bottom of the neck goes ~0.35 neck-widths below the hole centre
    fv = views[0]
    neck_sink = 0.18 * fv["neck_w"]
    for v in views:
        bx, by = v["bottom"]
        v["origin"] = (neck_base[0] - bx, neck_base[1] + neck_sink - by)
    head_pivot = (neck_base[0], fv["origin"][1] + fv["chin_y"])
    # cross-view sanity: the turnaround shares an eye line, so eyes far off the front view's eye line are false hits
    if fv["eyes"]:
        fy = np.mean([e["cy"] for e in fv["eyes"]]) + fv["origin"][1]
        hh_ = fv["rgba"].shape[0]
        for v in views[1:]:
            keep = [e for e in v["eyes"] if abs(e["cy"] + v["origin"][1] - fy) < 0.12 * hh_]
            if len(keep) != len(v["eyes"]): report.setdefault("dropped_eyes", {})[v["name"]] = len(v["eyes"]) - len(keep)
            v["eyes"] = keep
    neck_bone = (neck_base[0], neck_base[1] - 0.05 * fv["neck_w"])

    # ---------------- bones (bind: world positions, zero rotation) ----------------
    th_r = ty1 - ty0
    belt_y = ty1 - 0.17 * th_r
    chest_y = shoulder_y + 0.12 * th_r
    spine_y = (belt_y + chest_y) / 2
    bones = []
    def B(name, parent, x, y, **kw):
        bones.append({"name": name, "parent": parent, "x": round(float(x), 2), "y": round(float(y), 2), **kw})
    B("root", None, tcx, belt_y)
    B("hips", "root", tcx, belt_y)
    B("spine", "hips", tcx, spine_y)
    B("chest", "spine", tcx, chest_y)
    B("neck", "chest", *neck_bone)
    B("head", "neck", *head_pivot)
    for side in ("r", "l"):
        ai = arm_info[side]
        cl = (neck_base[0] + (ai["shoulder"][0] - neck_base[0]) * 0.35, shoulder_y + 0.02 * th_r)
        B("shoulder_" + side, "chest", *cl)
        B("upper_arm_" + side, "shoulder_" + side, *ai["shoulder"])
        B("lower_arm_" + side, "upper_arm_" + side, *ai["elbow"])
        B("hand_" + side, "lower_arm_" + side, *ai["wrist"])
    if tie_info:
        (kx, ky), (bx_, by_) = tie_info["top"], tie_info["bottom"]
        kyc = ky + 0.06 * (by_ - ky)
        B("tie_0", "chest", kx, kyc, secondary=True)
        B("tie_1", "tie_0", kx + (bx_ - kx) * 0.5, kyc + (by_ - kyc) * 0.5, secondary=True)
    bmap = {b["name"]: b for b in bones}

    # ---------------- weights ----------------
    def torso_w(x, y):
        t = (belt_y - y) / (belt_y - chest_y)
        if t <= 0: w = [["hips", 1]]
        elif t < 0.5: k = float(smooth(t / 0.5)); w = [["hips", 1 - k], ["spine", k]]
        elif t < 1: k = float(smooth((t - 0.5) / 0.5)); w = [["spine", 1 - k], ["chest", k]]
        else: w = [["chest", 1]]
        for side in ("r", "l"):
            ai = arm_info[side]; d = math.hypot(x - ai["shoulder"][0], y - ai["shoulder"][1])
            k = float(1 - smooth((d - ai["w"] * 0.3) / (ai["w"] * 1.1)))
            if k > 0.01:
                w = [[b, ww * (1 - k)] for b, ww in w] + [["shoulder_" + side, k]]
        return norm_w(w)
    tie_wmap = None
    if tie_info:
        tw_ = ndi.gaussian_filter(ndi.binary_dilation(tie, iterations=3).astype(float), 4)
        tie_wmap = np.clip(tw_ * 1.6, 0, 1)
        t0b, t1b = bmap["tie_0"], bmap["tie_1"]; tip_y = tie_info["bottom"][1]
    def torso_w_full(x, y):
        w = torso_w(x, y)
        if tie_wmap is not None:
            ix, iy = int(round(x - tx0)), int(round(y - ty0))
            k = float(tie_wmap[min(th - 1, max(0, iy)), min(tw - 1, max(0, ix))]) if 0 <= ix < tw + 2 and 0 <= iy < th + 2 else 0
            if k > 0.01 and y > t0b["y"] - 6:
                f = (y - t0b["y"]) / max(1, tip_y - t0b["y"])
                k *= float(smooth((y - t0b["y"] + 6) / 30))
                tw2 = [["tie_0", 1 - float(smooth((f - 0.35) / 0.4))], ["tie_1", float(smooth((f - 0.35) / 0.4))]]
                w = [[b, ww * (1 - k)] for b, ww in w] + [[b, ww * k] for b, ww in tw2]
        return norm_w(w)
    def arm_w(side):
        ai = arm_info[side]; sx, sy = ai["shoulder"]; wx, wy = ai["wrist"]
        L2 = (wx - sx) ** 2 + (wy - sy) ** 2
        def f(x, y):
            s = ((x - sx) * (wx - sx) + (y - sy) * (wy - sy)) / L2
            k = float(smooth((s - 0.40) / 0.18))
            return norm_w([["upper_arm_" + side, 1 - k], ["lower_arm_" + side, k]])
        return f
    def head_w(v):
        oy = v["origin"][1]; chin = oy + v["chin_y"]; bot = oy + v["bottom"][1]
        def f(x, y):
            k = float(smooth((bot - 0.12 * (bot - chin) - y) / max(1, (bot - chin) * 0.95)))
            return norm_w([["neck", 1 - k], ["head", k]])
        return f

    # ---------------- textures + meshes ----------------
    attachments = {}
    def save_tex(name, rgba):
        p = os.path.join(tex, name + ".png"); segment.save_rgba(rgba, p); return "tex/" + name + ".png"
    torso_rgba = torso["rgba"].copy()
    a = torso_rgba[..., 3].astype(float)
    a[hole] = 0; cb_rgba = torso_rgba.copy(); cb_rgba[..., 3] = np.where(collar_back, cb_rgba[..., 3], 0)
    a[collar_back] = 0; torso_rgba[..., 3] = a.astype(np.uint8)
    step = cfg.get("mesh_step", 12)
    attachments["torso"] = {"type": "mesh", "image": save_tex("torso", torso_rgba), "size": [tw, th], "origin": [tx0, ty0], **grid_mesh(torso_rgba[..., 3] / 255.0, step, (tx0, ty0), torso_w_full, (tw, th))}
    if collar_back.any():
        attachments["collar_back"] = {"type": "mesh", "image": save_tex("collar_back", cb_rgba), "size": [tw, th], "origin": [tx0, ty0], **grid_mesh(cb_rgba[..., 3] / 255.0, step, (tx0, ty0), torso_w, (tw, th))}
    if hips is not None:
        # trousers on the sheet are cut by the image edge: extend the legs downward by repeating the
        # lowest clean band of rows so the framing never shows where the art stops
        hr = hips["rgba"]; a_ = hr[..., 3] > 127; rows_on = np.nonzero(a_.any(1))[0]
        if cfg.get("extend_legs", 520) and hips["bbox"][3] >= SH - 3:
            last = rows_on.max() - 3; band = hr[last - 6:last]
            ext = np.concatenate([band] * int(math.ceil(cfg.get("extend_legs", 520) / 6)), 0)
            hr = np.concatenate([hr[:last], ext], 0)
            hips = {**hips, "rgba": hr}
        hx0, hy0 = hips["bbox"][:2]; hh, hw_ = hips["rgba"].shape[:2]
        # trousers top hides behind the jacket at belt height
        hox, hoy = tcx - (hips["centroid"][0] - hx0), belt_y - 6
        attachments["hips"] = {"type": "mesh", "image": save_tex("hips", hips["rgba"]), "size": [hw_, hh], "origin": [round(hox,2), round(hoy,2)], **grid_mesh(hips["rgba"][..., 3] / 255.0, 16, (hox, hoy), lambda x, y: [["hips", 1.0]], (hw_, hh))}
    for side in ("r", "l"):
        ai = arm_info[side]; rgba = ai["comp"]["rgba"]; hh, ww = rgba.shape[:2]
        attachments["arm_" + side] = {"type": "mesh", "image": save_tex("arm_" + side, rgba), "size": [ww, hh], "origin": [round(ai["origin"][0],2), round(ai["origin"][1],2)], **grid_mesh(rgba[..., 3] / 255.0, 10, ai["origin"], arm_w(side), (ww, hh))}
    for v in views:
        rgba = v["rgba"]; hh, ww = rgba.shape[:2]
        img = save_tex("head_" + v["name"], rgba)
        # eye mask texture (both painted eye interiors) for clipped procedural eyes
        em = np.zeros((hh, ww), bool)
        for e in v["eyes"]: em |= e["mask"]
        em = ndi.binary_dilation(em, iterations=2)
        ema = ndi.gaussian_filter(em.astype(float), 0.8)
        rgba_m = np.zeros((hh, ww, 4), np.uint8); rgba_m[..., :3] = 255; rgba_m[..., 3] = (np.clip(ema, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgba_m, "RGBA").save(os.path.join(tex, "eyes_" + v["name"] + ".png"))
        m = grid_mesh(rgba[..., 3] / 255.0, 10, v["origin"], head_w(v), (ww, hh))
        # parallax depth per vertex: 0 at the silhouette, 1 deep inside (features slide on turns)
        ed = ndi.distance_transform_edt(rgba[..., 3] > 127); dn = ed / max(1, ed.max())
        m["depth"] = [round(float(dn[min(hh - 1, int(y - v["origin"][1])), min(ww - 1, int(x - v["origin"][0]))]), 3) for x, y in m["vertices"]]
        attachments["head_" + v["name"]] = {"type": "mesh", "image": img, "origin": [round(v["origin"][0],2), round(v["origin"][1],2)], "eye_mask": "tex/eyes_" + v["name"] + ".png", "size": [ww, hh], **m}

    # ---------------- hands ----------------
    hcs, _ = segment.components(os.path.join(ROOT, S["hands"]))
    # 2-col grid: cluster rows by centroid y
    hcs = sorted(hcs, key=lambda c: c["centroid"][1])
    rows_, cur = [], []
    for c in hcs:
        if cur and c["centroid"][1] - np.mean([k["centroid"][1] for k in cur]) > 120: rows_.append(cur); cur = []
        cur.append(c)
    rows_.append(cur)
    ordered = [c for r in rows_ for c in sorted(r, key=lambda c: c["centroid"][0])]
    layout = cfg["hands_layout"]
    assert len(ordered) >= len(layout), "hand sheet: expected %d poses, found %d" % (len(layout), len(ordered))
    hand_poses = {}
    wrist_ws = []
    for spec, comp in zip(layout, ordered):
        a = comp["rgba"][..., 3] / 255.0; hh, ww = a.shape
        m = a > 0.5; ys, xs = np.nonzero(m)
        side_ = spec["wrist"]
        if side_ == "top": band = ys < ys.min() + 14; wp = (xs[band].mean(), ys.min() + 3); ww_ = np.ptp(xs[band])
        elif side_ == "bottom": band = ys > ys.max() - 14; wp = (xs[band].mean(), ys.max() - 3); ww_ = np.ptp(xs[band])
        elif side_ == "right": band = xs > xs.max() - 14; wp = (xs.max() - 3, ys[band].mean()); ww_ = np.ptp(ys[band])
        else: band = xs < xs.min() + 14; wp = (xs.min() + 3, ys[band].mean()); ww_ = np.ptp(ys[band])
        cx_, cy__ = xs.mean(), ys.mean()
        ang = math.degrees(math.atan2(cy__ - wp[1], cx_ - wp[0]))     # wrist -> fingers direction in texture
        wrist_ws.append(ww_)
        hand_poses[spec["pose"]] = {"comp": comp, "wrist": wp, "dir": ang, "wrist_w": float(ww_)}
    cuff_w = np.mean([arm_info[s]["cuff_w"] for s in arm_info])
    hscale = float(cuff_w * 0.78 / np.median(wrist_ws))
    report["hand_scale"] = round(hscale, 4)
    hands = {}
    for side in ("r", "l"):
        hb = bmap["hand_" + side]; hands[side] = {}
        for pose, hp in hand_poses.items():
            rgba = resize_rgba(hp["comp"]["rgba"], hscale)
            hh, ww = rgba.shape[:2]
            wx, wy = hp["wrist"][0] * hscale, hp["wrist"][1] * hscale
            mirror = side == "l"
            img = save_tex("hand_%s" % pose, rgba) if side == "r" else "tex/hand_%s.png" % pose
            # place texture so the wrist sits on the hand bone and fingers point along the bind forearm (down, 90deg)
            # we bake a local transform (rotation about the wrist) into the attachment: rotate fingers-dir -> 90deg
            d = hp["dir"] if not mirror else 180 - hp["dir"]
            rot = 90 - d
            wxm = (ww - wx) if mirror else wx
            ox, oy = hb["x"] - wxm, hb["y"] - wy
            mesh = grid_mesh(rgba[..., 3] / 255.0, 14, (ox, oy), lambda x, y, s=side: [["hand_" + s, 1.0]], (ww, hh), mirror=False)
            # mirror geometry for the left hand (flip about the wrist x) and rotate about the wrist
            cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            vs = []
            for (x, y), (u, v) in zip(mesh["vertices"], mesh["uvs"]):
                if mirror: x = hb["x"] - (x - (ox + wx)) ; 
                dx, dy = x - hb["x"], y - hb["y"]
                vs.append([round(hb["x"] + dx * cr - dy * sr, 2), round(hb["y"] + dx * sr + dy * cr, 2)])
            mesh["vertices"] = vs
            if mirror:  # keep triangle winding irrelevant (no culling); uvs unchanged since geometry mirrored
                pass
            hands[side][pose] = "hand_%s_%s" % (side, pose)
            # grip point = centroid of the half of the hand farthest from the wrist (rotated like the mesh)
            a_ = rgba[..., 3] > 127; ys_, xs_ = np.nonzero(a_)
            ddx, ddy = xs_ - wx, ys_ - wy; dist = np.hypot(ddx, ddy); far = dist > np.median(dist)
            gx, gy = float(ddx[far].mean()), float(ddy[far].mean())
            if mirror: gx = -gx
            grip = [round(gx * cr - gy * sr, 2), round(gx * sr + gy * cr, 2)]
            attachments["hand_%s_%s" % (side, pose)] = {"type": "mesh", "image": img, "size": [ww, hh], "grip": grip, **mesh}

    # ---------------- face description per view ----------------
    face_views = []
    yaw_of = cfg.get("view_yaw", {"front": 0.0, "q_left": -0.5, "profile_left": -1.0})
    for v in views:
        ox, oy = v["origin"]
        eyes = [{"cx": round(ox + e["cx"], 2), "cy": round(oy + e["cy"], 2), "rx": round(e["rx"], 2), "ry": round(e["ry"], 2),
                 "color": e["color"]} for e in v["eyes"]]
        mth = None
        if v["mouth"]:
            mo = v["mouth"]; c = mo["coef"]
            # convert polynomial from texture x to rig x: y = a(x-ox)^2 + b(x-ox) + c + oy
            a_, b_, c_ = c
            A = a_; Bc = b_ - 2 * a_ * ox; C = a_ * ox * ox - b_ * ox + c_ + oy
            yaw = yaw_of.get(v["name"], 0)
            depth = (oy + v["chin_y"] - (mo["cy"] + oy)) * 0.62
            mth = {"coef": [A, Bc, C], "x0": round(ox + mo["x0"], 2), "x1": round(ox + mo["x1"], 2), "jaw_depth": round(depth, 2), "fill": mo.get("fill"), "transferred": bool(mo.get("transferred")),
                   "hinge": "both" if abs(yaw) < 0.25 else ("right" if yaw < 0 else "left")}
        face_views.append({"view": v["name"], "attachment": "head_" + v["name"], "yaw": yaw_of.get(v["name"], 0), "eyes": eyes,
                           "mouth": mth, "skin": v["skin"], "ink": v["ink"], "origin": [round(ox, 2), round(oy, 2)],
                           "detector": "mediapipe" if v["mediapipe"] else "cartoon-blob"})
        report["landmarks"]["view_" + v["name"]] = {"eyes": len(eyes), "mouth": mth is not None, "detector": face_views[-1]["detector"]}
    # mirrored views
    for fvw in list(face_views):
        if fvw["yaw"] < 0:
            face_views.append({"view": fvw["view"].replace("left", "right"), "mirror_of": fvw["view"], "yaw": -fvw["yaw"]})

    # ---------------- slots (draw order, back to front) ----------------
    slots = []
    if "collar_back" in attachments: slots.append({"name": "collar_back", "bone": "chest", "attachment": "collar_back"})
    slots.append({"name": "head", "bone": "head", "attachment": "head_" + views[0]["name"], "kind": "head_views"})
    if "hips" in attachments: slots.append({"name": "hips", "bone": "hips", "attachment": "hips"})
    slots.append({"name": "torso", "bone": "chest", "attachment": "torso"})
    for side in ("l", "r"):  # far-ish arm first; both arms in front of the torso
        slots.append({"name": "hand_" + side, "bone": "hand_" + side, "attachment": hands[side].get("rest"), "kind": "hand"})
        slots.append({"name": "arm_" + side, "bone": "upper_arm_" + side, "attachment": "arm_" + side})

    anchors = {
        "tie_knot": [round(bmap["tie_0"]["x"], 1), round(bmap["tie_0"]["y"] + 10, 1), "chest"] if tie_info else [round(neck_base[0], 1), round(neck_base[1] + 40, 1), "chest"],
        "chest_center": [round(tcx, 1), round(chest_y + 0.08 * th_r, 1), "chest"],
        "belly": [round(tcx, 1), round(belt_y - 0.12 * th_r, 1), "spine"],
        "chin": [round(head_pivot[0], 1), round(head_pivot[1] - 10, 1), "head"],
        "lapel_r": [round(tcx - 0.2 * (xr - xl), 1), round(chest_y + 0.05 * th_r, 1), "chest"],
        "lapel_l": [round(tcx + 0.2 * (xr - xl), 1), round(chest_y + 0.05 * th_r, 1), "chest"],
    }
    unit = float(arm_info["l"]["shoulder"][0] - arm_info["r"]["shoulder"][0])
    rig = {
        "schema": SCHEMA_VERSION, "slug": slug, "style": cfg.get("style", "pogtown-2d-v1"),
        "unit": round(unit, 2), "size_hint": {"top": float(min(v["origin"][1] for v in views)), "bottom": float(ty1 + 200),
                      "art_bottom": float(attachments["hips"]["origin"][1] + attachments["hips"]["size"][1]) if "hips" in attachments else float(ty1)},
        "skeleton": {"naming": "oddrig-humanoid/1", "bones": bones},
        "slots": slots, "attachments": attachments,
        "hands": {"set": list(hand_poses), "by_side": hands},
        "face": {"views": face_views,
                 "eyes": {"type": cfg.get("eyes", {}).get("type", "ball"), **cfg.get("eyes", {})},
                 "mouth": {"type": cfg.get("mouth", {}).get("type", "jaw"), **cfg.get("mouth", {})},
                 "visemes": ["X", "A", "B", "C", "D", "E", "F", "G", "H"],
                 "expressions": ["neutral", "deadpan", "annoyed", "surprised", "confused", "smug", "grin", "self_satisfied", "confident", "suspicious", "embarrassed", "sad"]},
        "anchors": anchors,
        "physics": cfg.get("physics", [{"bone": "tie_0", "stiffness": 38, "damping": 5.5, "gain": 0.9}, {"bone": "tie_1", "stiffness": 30, "damping": 4.5, "gain": 1.2}] if tie_info else []),
        "limits": cfg.get("limits", {}),
        "clips": {"inherit": "shared", "overrides": cfg.get("clip_overrides", {})},
        "provenance": {"sheets": S, "onboard": "rigkit/build_rig.py", "report": report},
    }
    json.dump(rig, open(os.path.join(out, "rig.json"), "w"), separators=(",", ":"))
    validate_png(rig, out, views, arm_info, hips)
    print(json.dumps(report, indent=1))
    return rig


def validate_png(rig, out, views, arm_info, hips):
    """Bind-pose composite with joints, weights hint and face features (a validation still)."""
    W, H = 900, 1500
    canvas = Image.new("RGBA", (W, H), (40, 40, 52, 255))
    ox, oy = 0, 0
    order = [s["attachment"] for s in rig["slots"]]
    for name in order:
        at = rig["attachments"].get(name)
        if not at: continue
        im = Image.open(os.path.join(out, at["image"])).convert("RGBA")
        if name.startswith("hand_"):
            continue
        x0, y0 = at["origin"]
        canvas.alpha_composite(im, (int(round(x0)), int(round(y0))))
    d = ImageDraw.Draw(canvas)
    bm = {b["name"]: b for b in rig["skeleton"]["bones"]}
    for b in rig["skeleton"]["bones"]:
        if b["parent"]:
            p = bm[b["parent"]]; d.line([(p["x"], p["y"]), (b["x"], b["y"])], fill=(255, 220, 0, 255), width=3)
    for b in rig["skeleton"]["bones"]:
        d.ellipse([b["x"] - 6, b["y"] - 6, b["x"] + 6, b["y"] + 6], fill=(255, 60, 60, 255))
    fvw = rig["face"]["views"][0]
    for e in fvw["eyes"]:
        d.ellipse([e["cx"] - e["rx"], e["cy"] - e["ry"], e["cx"] + e["rx"], e["cy"] + e["ry"]], outline=(0, 255, 255, 255), width=2)
    if fvw["mouth"]:
        A, Bc, C = fvw["mouth"]["coef"]
        pts = [(x, A * x * x + Bc * x + C) for x in np.linspace(fvw["mouth"]["x0"], fvw["mouth"]["x1"], 30)]
        d.line(pts, fill=(0, 255, 0, 255), width=3)
    canvas.convert("RGB").save(os.path.join(out, "validate.png"))


if __name__ == "__main__":
    build(sys.argv[1])
