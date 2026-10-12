"""Two-hander on a shared stage. duet.json:
{"title":..., "cast": {"A": "<slug>", "B": "<slug>"}, "lines": [{"who":"A","id":..,"type":..,"text":..,"performance":{..},"pause_after_ms":..}, ...]}
Each line is a delivery beat for its speaker; the listener gets listening behaviour (looks at the
speaker, small nods, a reaction expression on the partner's punchline) and the camera cuts
wide <-> speaker close-up on punchlines/closers.
"""
import json, os, sys, time
import numpy as np, soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import run as R  # noqa
import tts, director, render  # noqa

SLOTS = {"A": {"x": -150, "y": 560, "scale": 0.98, "side": 1, "look": [0.45, 0]},
         "B": {"x": 500, "y": 560, "scale": 0.98, "side": -1, "look": [-0.45, 0]}}

def run(a):
    sc = json.load(open(a.script))
    name = sc.get("name", "duet")
    out = os.path.join(ROOT, "out", name); os.makedirs(out, exist_ok=True)
    t_all = time.time()
    cast = {}
    for k, slug in sc["cast"].items():
        ch, _, _ = R.load_graph(slug); ch["_slug"] = slug
        rig, layers, rigdir = R.load_rig(slug)
        cast[k] = {"slug": slug, "ch": ch, "rig": rig, "layers": layers, "dir": rigdir}
    # synth every line with its speaker's voice
    items = []; labels = set()
    for i, ln in enumerate(sc["lines"]):
        c = cast[ln["who"]]; b = dict(ln); b.setdefault("id", "l%d" % i)
        wav = os.path.join(out, "line_%02d_%s.wav" % (i, ln["who"]))
        if not os.path.exists(wav + ".json"):
            words, label = R.synth_beat(c["ch"], b, wav)
            json.dump({"words": words, "label": label}, open(wav + ".json", "w"))
        meta = json.load(open(wav + ".json")); labels.add(meta["label"])
        hold = c["ch"]["profile"].get("punchline_hold_ms", 1000) / 1000.0
        p = b.get("pause_after_ms", 250) / 1000.0
        if b.get("type") in ("punchline", "closer"): p = max(p, hold)
        items.append({"beat": b, "wav": wav, "words": meta["words"], "dur": tts.duration(wav), "pause": p, "who": ln["who"]})
    wavp = os.path.join(out, "performance.wav")
    timing, dur = R.assemble(items, wavp)
    # per-speaker audio for per-speaker mouth cues + envelopes
    sr = tts.SR; full, _ = sf.read(wavp)
    perf = []
    shots = [{"t": 0, "zoom": 1.0, "fx": 540, "fy": 900}]
    for k, c in cast.items():
        mine = [i for i, it in enumerate(items) if it["who"] == k]
        solo = np.zeros_like(full)
        for i in mine:
            s0 = int(timing[i]["t0"] * sr); s1 = int(timing[i]["t1"] * sr); solo[s0:s1] = full[s0:s1]
        sw = os.path.join(out, "solo_%s.wav" % k); sf.write(sw, solo, sr)
        text = " ".join(items[i]["beat"]["text"] for i in mine)
        cues, _ = R.rhubarb(sw, text, os.path.join(out, "mouth_%s.json" % k))
        ev = director.direct(c["ch"], [items[i]["beat"] for i in mine], [timing[i] for i in mine], seed=7 + len(perf))
        # listening behaviour while the partner speaks
        other = [i for i, it in enumerate(items) if it["who"] != k]
        for i in other:
            tm = timing[i]; b = items[i]["beat"]
            ev["looks"].append({"t": round(tm["t0"] + 0.15, 3), "x": SLOTS[k]["look"][0] * 1.6, "y": 0.05, "why": "listen"})
            ev["nods"].append({"t": round(tm["t0"] + (tm["t1"] - tm["t0"]) * 0.6, 3), "amp": 0.5})
            if b.get("type") == "punchline":
                ev["expr"].append({"t": round(tm["t1"] + 0.1, 3), "expr": "deadpan" if c["ch"].get("vibe") == "deadpan" else "smug", "beat": "react"})
                ev["looks"].append({"t": round(tm["t1"] + 0.35, 3), "x": 0, "y": 0, "why": "react to camera"})
        for i in mine:  # speaker addresses partner at the start of a line, audience by the end
            tm = timing[i]
            ev["looks"].append({"t": round(tm["t0"] - 0.05, 3), "x": SLOTS[k]["look"][0], "y": 0, "why": "address partner"})
            ev["looks"].append({"t": round(tm["t0"] + (tm["t1"] - tm["t0"]) * 0.55, 3), "x": 0, "y": 0, "why": "to camera"})
        ev["looks"].sort(key=lambda x: x["t"]); ev["expr"].sort(key=lambda x: x["t"])
        sl = SLOTS[k]
        place = {"scale": sl["scale"], "x": sl["x"], "y": sl["y"]}
        perf.append(R.performer_block(c["slug"], c["rig"], c["layers"], c["dir"], ev, cues, R.envelope(sw), side=sl["side"], place=place))
    # camera: cut to speaker close-up for punchlines/closers, back to wide after the hold
    for i, it in enumerate(items):
        if it["beat"].get("type") in ("punchline", "closer"):
            sl = SLOTS[it["who"]]; cx = sl["x"] + 384 * sl["scale"]; cy = sl["y"] + 330 * sl["scale"]
            shots.append({"t": round(timing[i]["t0"] - 0.05, 3), "zoom": 1.55, "fx": cx, "fy": cy})
            shots.append({"t": round(timing[i]["hold_end"] - 0.1, 3), "zoom": 1.0, "fx": 540, "fy": 900})
    shots.sort(key=lambda s: s["t"])
    tl = {"fps": R.FPS, "duration": round(dur, 3), "background": "/assets/stage/office.png", "fadeIn": 0.25, "bgScale": 1.75,
          "captions": R.captions(timing), "shots": shots,
          "tags": [{"t0": 0.3, "t1": 2.6, "name": cast["A"]["rig"]["display_name"], "sub": cast["A"]["rig"]["tagline"], "color": cast["A"]["rig"]["tag_color"], "x": 50, "y": 140},
                   {"t0": 0.6, "t1": 2.9, "name": cast["B"]["rig"]["display_name"], "sub": cast["B"]["rig"]["tagline"], "color": cast["B"]["rig"]["tag_color"], "x": 50, "y": 290}],
          "performers": perf, "meta": {"tts": sorted(labels), "lines": len(items)}}
    tlp = os.path.join(out, "timeline.json"); json.dump(tl, open(tlp, "w"), indent=1)
    if a.stills:
        print(render.still(tlp, a.stills, os.path.join(out, "still.jpg"))); return
    mp4 = os.path.join(out, "%s_%ds.mp4" % (name, int(round(dur))))
    st = render.render(tlp, wavp, mp4, workers=a.workers)
    rep = {"mp4": mp4, "duration_s": round(dur, 2), "tts": sorted(labels), **st, "total_s": round(time.time() - t_all, 1)}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))
