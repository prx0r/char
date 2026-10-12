"""Two-hander on the OddRig stack. Same script format as duet.py:
{"name":..., "cast": {"A": "<slug>", "B": "<slug>"}, "lines": [{"who":"A","type":..,"text":..,"performance":{..},"pause_after_ms":..}, ...]}

Each speaker's lines go through director2 (shared clips); the listener gets listening behaviour from the same
library: gaze to the partner (the head spring turns into the 3/4 view), nods, and a reaction clip on the
partner's punchline. Camera cuts wide <-> speaker close-up on punchlines/closers.
"""
import json, os, time
import numpy as np, soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import run as R  # noqa
import tts, director2, render  # noqa

SLOTS = {"A": {"x": 300, "toward": 1}, "B": {"x": 790, "toward": -1}}


def run(a):
    sc = json.load(open(a.script))
    name = sc.get("name", "duet") + "-oddrig"
    out = os.path.join(ROOT, "out", name); os.makedirs(out, exist_ok=True)
    t_all = time.time()
    cast = {}
    for k, slug in sc["cast"].items():
        ch, _, _ = R.load_graph(slug); ch["_slug"] = slug
        rig = R.ensure_rig(slug)
        cast[k] = {"slug": slug, "ch": ch, "rig": rig, "meta": R.rig_meta(slug)}
    items = []; labels = set()
    for i, ln in enumerate(sc["lines"]):
        c = cast[ln["who"]]; b = dict(ln); b.setdefault("id", "l%d" % i)
        wav = os.path.join(out, "line_%02d_%s.wav" % (i, ln["who"]))
        if not os.path.exists(wav + ".json"):
            words, label = R.synth_beat(c["ch"], b, wav)
            json.dump({"words": words, "label": label}, open(wav + ".json", "w"))
        meta = json.load(open(wav + ".json")); labels.add(meta["label"])
        hold = c["ch"].get("profile", {}).get("punchline_hold_ms", 1000) / 1000.0
        p = b.get("pause_after_ms", 250) / 1000.0
        if b.get("type") in ("punchline", "closer"): p = max(p, hold)
        items.append({"beat": b, "wav": wav, "words": meta["words"], "dur": tts.duration(wav), "pause": p, "who": ln["who"]})
    wavp = os.path.join(out, "performance.wav")
    timing, dur = R.assemble(items, wavp)
    sr = tts.SR; full, _ = sf.read(wavp)
    perf = []; shots = [{"t": 0, "zoom": 1.0, "fx": 540, "fy": 820}]
    for k, c in cast.items():
        mine = [i for i, it in enumerate(items) if it["who"] == k]
        solo = np.zeros_like(full)
        for i in mine:
            s0 = int(timing[i]["t0"] * sr); s1 = int(timing[i]["t1"] * sr); solo[s0:s1] = full[s0:s1]
        sw = os.path.join(out, "solo_%s.wav" % k); sf.write(sw, solo, sr)
        cues, _ = R.rhubarb(sw, " ".join(items[i]["beat"]["text"] for i in mine), os.path.join(out, "mouth_%s.json" % k))
        ev = director2.direct(c["ch"], [items[i]["beat"] for i in mine], [timing[i] for i in mine], seed=7 + len(perf))
        tw = SLOTS[k]["toward"]
        for i in [i for i, it in enumerate(items) if it["who"] != k]:      # listening
            tm = timing[i]; b = items[i]["beat"]
            ev["looks"].append({"t": round(tm["t0"] + 0.15, 3), "x": 0.85 * tw, "y": 0.05, "why": "listen"})
            ev["clips"].append({"clip": "nod", "t0": round(tm["t0"] + (tm["t1"] - tm["t0"]) * 0.6, 3), "amp": 0.5, "why": "listen-nod"})
            if b.get("type") == "punchline":
                ev["clips"].append({"clip": "head_tilt", "t0": round(tm["t1"] + 0.05, 3), "dur": 1.6, "why": "react"})
                ev["expr"].append({"t": round(tm["t1"] + 0.1, 3), "expr": "deadpan", "beat": "react"})
                ev["looks"].append({"t": round(tm["t1"] + 0.45, 3), "x": 0, "y": 0, "why": "react to camera"})
        for i in mine:                                                       # address partner, then the audience
            tm = timing[i]
            ev["looks"].append({"t": round(tm["t0"] - 0.05, 3), "x": 0.7 * tw, "y": 0, "why": "address partner"})
            ev["looks"].append({"t": round(tm["t0"] + (tm["t1"] - tm["t0"]) * 0.55, 3), "x": 0, "y": 0, "why": "to camera"})
        ev["looks"].sort(key=lambda x: x["t"]); ev["expr"].sort(key=lambda x: x["t"]); ev["clips"].sort(key=lambda x: x["t0"])
        place = R.place_for(c["rig"], x=SLOTS[k]["x"], head_top=520, shoulder_frac=0.25)
        c["place"] = place
        perf.append(R.performer_v2(c["slug"], c["rig"], ev, cues, R.envelope(sw), place, c["ch"]))
    for i, it in enumerate(items):
        if it["beat"].get("type") in ("punchline", "closer"):
            pl = cast[it["who"]]["place"]
            shots.append({"t": round(timing[i]["t0"] - 0.05, 3), "zoom": 1.45, "fx": pl["x"], "fy": 700})
            shots.append({"t": round(timing[i]["hold_end"] - 0.1, 3), "zoom": 1.0, "fx": 540, "fy": 820})
    shots.sort(key=lambda s: s["t"])
    tags = []
    for j, k in enumerate(("A", "B")):
        m = cast[k]["meta"]
        tags.append({"t0": 0.3 + 0.3 * j, "t1": 2.6 + 0.3 * j, "name": m.get("display_name", cast[k]["ch"]["name"].split(",")[0]), "sub": m.get("tagline", ""), "color": m.get("tag_color", "#7ed957"), "x": 50, "y": 140 + 150 * j})
    tl = {"page": "stage2.html", "fps": R.FPS, "duration": round(dur, 3), "background": "/assets/stage/office.png", "fadeIn": 0.25,
          "captions": R.captions(timing), "shots": shots, "tags": tags, "performers": perf, "labels": R.voice_labels(labels),
          "meta": {"tts": sorted(labels), "lines": len(items), "stack": "oddrig/1"}}
    tlp = os.path.join(out, "timeline.json"); json.dump(tl, open(tlp, "w"), indent=1)
    if a.stills:
        print(render.still(tlp, a.stills, os.path.join(out, "still.jpg"))); return
    mp4 = os.path.join(out, "%s_%ds.mp4" % (name, int(round(dur))))
    st = render.render(tlp, wavp, mp4, workers=a.workers)
    rep = {"mp4": mp4, "duration_s": round(dur, 2), "tts": sorted(labels), **st, "total_s": round(time.time() - t_all, 1)}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))
