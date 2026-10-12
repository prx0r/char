#!/usr/bin/env python3
"""Character -> performance pipeline.

  python3 run.py perform --character reptilian-ralph-middle-management [--seconds 15]
  python3 run.py add-character --from-subject chris --photo <path>
  python3 run.py duet --a reptilian-ralph-middle-management --b chris-prior-golf-dad --script <duet.json>
  python3 run.py still --character <slug> --t 1.0 3.2       (inspect frames)
"""
import argparse, json, os, subprocess, sys, time, shutil
import numpy as np, soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import tts, director, render  # noqa
import director2  # noqa

CHARS_REPO = os.environ.get("CHARS_REPO", "/workspace/work/chars")
RHUBARB = os.path.join(ROOT, "vendors/rhubarb-lip-sync/build/rhubarb/rhubarb")
FPS = 30
LEAD_IN = 0.45
TAIL = 0.7

def load_graph(slug):
    """character.json: the chars repo pack first (source of truth: freaks/<slug>), else a char-lab copy for
    characters that only exist here. delivery.json: same order, independently (a lab-written delivery can
    accompany a graph that lives only in the chars repo)."""
    bases = (os.path.join(CHARS_REPO, "freaks", slug), os.path.join(ROOT, "characters", slug))
    cj = next((os.path.join(b, "character.json") for b in bases if os.path.exists(os.path.join(b, "character.json"))), None)
    if not cj: raise SystemExit("no character.json for " + slug)
    dj = next((os.path.join(b, "delivery.json") for b in bases if os.path.exists(os.path.join(b, "delivery.json"))), None)
    return json.load(open(cj)), (json.load(open(dj)) if dj else None), os.path.dirname(cj)

def load_set(path):
    """A pipeline set (chars freaks/<slug>/sets/*.json, human-set.v1) or any delivery-shaped JSON -> beats in
    delivery shape. The set's own beat types/pauses are kept; performance defaults follow the beat type."""
    d = json.load(open(path))
    perf = {"setup": ("neutral", "normal", 0.5), "escalation": ("neutral", "small gesture", 0.6), "punchline": ("deadpan", "hold still", 0.9),
            "tag": ("annoyed", "small gesture", 0.7), "closer": ("smug", "lean in", 1.0)}
    beats = []
    for i, b in enumerate(d["beats"]):
        e, g, em = perf.get(b.get("type"), perf["setup"])
        beats.append({"id": b.get("id", "s%d" % (i + 1)), "type": b.get("type", "setup"), "text": b["text"],
                      "performance": b.get("performance") or {"expression": e, "gesture": g, "look": "audience"},
                      "speech": b.get("speech") or {"emphasis": b.get("emphasis", em), "pace": "normal"},
                      "pause_after_ms": int(b.get("pause_after_ms", 300))})
    return {"beats": beats, "source": os.path.abspath(path), "scenario_id": d.get("scenario_id"), "run_id": d.get("run_id")}

def load_rig(slug):
    d = os.path.join(ROOT, "characters", slug)
    if not os.path.exists(os.path.join(d, "layers", "layers.json")):
        subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "prep_rig.py"), slug], check=True)
    return json.load(open(os.path.join(d, "rig.json"))), json.load(open(os.path.join(d, "layers", "layers.json"))), "/characters/" + slug

def synth_beat(character, beat, wav, voice_override=None):
    v = voice_override or character.get("voice", {})
    if v.get("provider") == "voxcpm" and os.environ.get("VOXCPM_OK") != "1":
        # VoxCPM2 (2B) is OOM-killed on this 8 GB CPU box; set VOXCPM_OK=1 on a bigger machine.
        v = rig_meta(character["_slug"]).get("voice_fallback", v)
    if v.get("provider") == "edge-tts" and os.environ.get("EDGE_TTS_OK") != "1":
        # edge-tts returns HTTP 403 from this sandbox (see experiments/CAPABILITY_REPORT.md);
        # use the rig's local stand-in voice and label it honestly.
        v = rig_meta(character["_slug"]).get("voice_fallback", v)
    prof = character.get("profile", {})
    pace = float(prof.get("pace", 1.0))
    if v.get("provider") == "voxcpm":
        ctrl = v["design"]
        _, words, label = tts.voxcpm(beat["text"], ctrl, wav, seed=int(v.get("seed", 7)))
        tts.trim_silence(wav)
        words = tts.align_words_by_energy(wav, beat["text"])
    elif v.get("provider") == "kokoro":
        _, words, label = tts.kokoro(beat["text"], v["voice_id"], wav, speed=float(v.get("speed", 1.0)) * (0.92 + 0.08 * pace), lang=v.get("lang", "en-us"))
    else:
        rate = "%+d%%" % int(round((pace - 1.0) * 100))
        _, words, label = tts.edge(beat["text"], v.get("voice_id", "en-US-ChristopherNeural"), wav, rate=rate)
        off = tts.trim_silence(wav)
        for w in words: w["t0"] = round(max(0, w["t0"] - off), 3); w["t1"] = round(max(0, w["t1"] - off), 3)
    return words, label

def _beat_audio(character, b, workdir, voice_override, labels):
    wav = os.path.join(workdir, "beat_%s.wav" % b["id"])
    if not os.path.exists(wav + ".json"):
        words, label = synth_beat(character, b, wav, voice_override)
        json.dump({"words": words, "label": label}, open(wav + ".json", "w"))
    meta = json.load(open(wav + ".json")); labels.add(meta["label"])
    return wav, meta, tts.duration(wav)

def select_and_synth(character, beats, workdir, max_s, voice_override=None, slack=2.5, trim=False):
    """Fit a ~max_s cut that ends on the closer: walk back contiguously from the closer; when the
    next beat no longer fits, pull in one earlier beat that does, preferring beats that fire a
    character action not yet used (on_groan etc.), then punchlines. Order is preserved."""
    hold = character.get("profile", {}).get("punchline_hold_ms", 1000) / 1000.0
    end = len(beats) - 1
    for i, b in enumerate(beats):
        if b.get("type") == "closer": end = i
    labels = set(); chosen = {}; total = LEAD_IN + TAIL
    def pause_of(b):
        p = b.get("pause_after_ms", 300) / 1000.0
        return max(p, hold) if b.get("type") in ("punchline", "closer") else p
    i = end
    while i >= 0:
        b = beats[i]; wav, meta, d = _beat_audio(character, b, workdir, voice_override, labels)
        cost = d + pause_of(b)
        if chosen and total + cost > max_s + slack: break
        chosen[i] = {"beat": b, "wav": wav, "words": meta["words"], "dur": d, "pause": pause_of(b)}; total += cost; i -= 1
    first = min(chosen)
    if total < max_s - 1.5 and first > 0:
        acts = character.get("actions") or {}
        def score(b):
            e = b.get("performance", {}).get("expression"); g = b.get("performance", {}).get("gesture")
            s = 0
            if (e == "annoyed" or g == "small gesture") and any(a.get("trigger") == "on_groan" for a in acts.values()): s += 3
            if b.get("type") == "punchline": s += 2
            if float(b.get("speech", {}).get("emphasis", 0.5)) >= 0.75 and any(a.get("trigger") == "on_emphasis" for a in acts.values()): s += 2
            return s
        placed = False
        for j in sorted(range(first - 1, -1, -1), key=lambda j: -score(beats[j])):
            b = beats[j]; wav, meta, d = _beat_audio(character, b, workdir, voice_override, labels)
            cost = d + pause_of(b)
            if total + cost <= max_s + slack:
                chosen[j] = {"beat": b, "wav": wav, "words": meta["words"], "dur": d, "pause": pause_of(b)}; total += cost; placed = True; break
        if not placed and trim:
            # no whole beat fits: use the opening sentences of the best-scoring earlier beat (verbatim prefix,
            # at least two sentences) so the cut still lands on ~max_s and keeps the action trigger
            import re
            for j in sorted(range(first - 1, -1, -1), key=lambda j: -score(beats[j])):
                if score(beats[j]) <= 0: continue
                sents = [x for x in re.split(r"(?<=[.?!])\s+", beats[j]["text"]) if x]
                for k in range(len(sents) - 1, 1, -1):
                    b = dict(beats[j]); b["text"] = " ".join(sents[:k]); b["id"] = "%s~%d" % (beats[j]["id"], k)
                    b["pause_after_ms"] = min(600, beats[j].get("pause_after_ms", 300))
                    wav, meta, d = _beat_audio(character, b, workdir, voice_override, labels)
                    cost = d + pause_of(b)
                    if total + cost <= max_s + slack:
                        chosen[j] = {"beat": b, "wav": wav, "words": meta["words"], "dur": d, "pause": pause_of(b)}; total += cost; placed = True; break
                if placed: break
    return [chosen[k] for k in sorted(chosen)], sorted(labels)

def assemble(chosen, out_wav, sr=tts.SR):
    parts = [np.zeros(int(LEAD_IN * sr))]; t = LEAD_IN; timing = []
    for c in chosen:
        a, r = sf.read(c["wav"]); assert r == sr
        t0 = t; parts.append(a); t += len(a) / sr
        timing.append({"t0": round(t0, 3), "t1": round(t, 3), "hold_end": round(t + c["pause"], 3),
                       "words": [{"w": w["w"], "t0": round(w["t0"] + t0, 3), "t1": round(w["t1"] + t0, 3)} for w in c["words"]]})
        parts.append(np.zeros(int(c["pause"] * sr))); t += c["pause"]
    parts.append(np.zeros(int(TAIL * sr)))
    a = np.concatenate(parts)
    a = a / max(1e-6, np.abs(a).max()) * 0.89
    sf.write(out_wav, a, sr)
    return timing, len(a) / sr

def rhubarb(wav, text, out_json):
    dlg = out_json + ".txt"; open(dlg, "w").write(text)
    t = time.time()
    r = subprocess.run([RHUBARB, "-q", "-f", "json", "-d", dlg, "--extendedShapes", "GHX", "-o", out_json, wav], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError("rhubarb: " + r.stderr[-800:])
    cues = json.load(open(out_json))["mouthCues"]
    return [{"t": round(c["start"], 3), "v": c["value"]} for c in cues], round(time.time() - t, 1)

def envelope(wav, fps=FPS):
    a, sr = sf.read(wav); hop = sr // fps
    n = len(a) // hop
    rms = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) for i in range(n)])
    rms = rms / max(1e-6, np.percentile(rms, 98))
    sm = np.convolve(np.clip(rms, 0, 1), np.ones(3) / 3, mode="same")
    return [round(float(x), 3) for x in sm]

def captions(timing, max_words=4):
    caps = []
    for tm in timing:
        ws = tm["words"]; i = 0
        while i < len(ws):
            chunk = ws[i:i + max_words]
            # break after sentence punctuation
            for j, w in enumerate(chunk):
                if w["w"][-1:] in ".?!" and j < len(chunk) - 1: chunk = chunk[:j + 1]; break
            caps.append({"t0": chunk[0]["t0"] - 0.05, "t1": chunk[-1]["t1"] + 0.12, "words": chunk}); i += len(chunk)
    for a, b in zip(caps, caps[1:]): a["t1"] = min(a["t1"], b["t0"] - 0.01)
    return caps

def performer_block(slug, rig, layers, rigdir, ev, cues, env, offset=(0, 0), side=1, place=None, base_look=None):
    r = json.loads(json.dumps(rig))
    if place: r["place"] = place
    return {"slug": slug, "dir": rigdir, "rig": r, "layers": layers, "events": ev, "mouth": cues, "env": env,
            "env_fps": FPS, "offset": list(offset), "side": side, "base_look": base_look}

def cmd_perform(a):
    slug = a.character
    character, delivery, gdir = load_graph(slug)
    character["_slug"] = slug
    rig, layers, rigdir = load_rig(slug)
    out = os.path.join(ROOT, "out", slug); os.makedirs(out, exist_ok=True)
    t_all = time.time()
    chosen, labels = select_and_synth(character, delivery["beats"], out, a.seconds)
    t_tts = time.time() - t_all
    wav = os.path.join(out, "performance.wav")
    timing, dur = assemble(chosen, wav)
    text = " ".join(c["beat"]["text"] for c in chosen)
    cues, t_rh = rhubarb(wav, text, os.path.join(out, "mouth.json"))
    ev = director.direct(character, [c["beat"] for c in chosen], timing, seed=a.seed)
    tl = {"fps": FPS, "duration": round(dur, 3), "background": rig.get("stage", "/assets/stage/office.png"), "fadeIn": 0.25,
          "captions": captions(timing),
          "tags": [{"t0": 0.3, "t1": 3.0, "name": rig.get("display_name", character["name"].split(",")[0]), "sub": rig.get("tagline", character.get("species", "")), "color": rig.get("tag_color", "#7ed957")}],
          "performers": [performer_block(slug, rig, layers, rigdir, ev, cues, envelope(wav))],
          "meta": {"beats": [c["beat"]["id"] for c in chosen], "tts": labels}}
    tlp = os.path.join(out, "timeline.json"); json.dump(tl, open(tlp, "w"), indent=1)
    if a.stills is not None and len(a.stills):
        print(render.still(tlp, a.stills, os.path.join(out, "still.jpg"))); return
    mp4 = os.path.join(out, "%s_%ds.mp4" % (slug, int(round(dur))))
    st = render.render(tlp, wav, mp4, workers=a.workers)
    rep = {"mp4": mp4, "duration_s": round(dur, 2), "beats": tl["meta"]["beats"], "tts": labels, "tts_s": round(t_tts, 1), "rhubarb_s": t_rh,
           "gestures": [(g["name"], g["t0"], g["why"]) for g in ev["gestures"]], **st, "total_s": round(time.time() - t_all, 1)}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

# ---------------- OddRig stack (v2) ----------------
def rig_path(slug):
    return os.path.join(ROOT, "characters", slug, "rig", "rig.json")

def ensure_rig(slug, rebuild=False):
    if rebuild or not os.path.exists(rig_path(slug)):
        subprocess.run([sys.executable, os.path.join(ROOT, "rigkit", "build_rig.py"), slug], check=True)
    return json.load(open(rig_path(slug)))

def place_for(rig, x=540, head_top=330, shoulder_frac=0.36, scale=None):
    """Frame waist-up: shoulders = shoulder_frac of the width, head top at head_top; then make sure the bottom
    of the art (where the drawing stops) is below the frame by moving down, then scaling up."""
    s = scale or (W_FRAME * shoulder_frac / rig["unit"])
    root = next(b for b in rig["skeleton"]["bones"] if b["name"] == "root")
    top = rig["size_hint"]["top"]; bottom = rig["size_hint"].get("art_bottom", rig["size_hint"]["bottom"])
    for _ in range(40):
        if head_top + (bottom - top) * s >= 1960: break
        if head_top < 470: head_top += 12
        else: s *= 1.02
    return {"x": x, "y": round(head_top + (root["y"] - top) * s, 1), "scale": round(s, 4)}

W_FRAME = 1080

def voice_labels(labels):
    """On-screen disclosure: which TTS spoke. Real-person packs never use a cloned voice."""
    if any(str(l).startswith("kokoro") for l in labels):
        return [{"text": "AI voice: Kokoro-82M preset stand-in, not the real person's voice"}]
    return [{"text": "AI voice: " + ", ".join(sorted(set(str(l).split(":")[0] for l in labels)))}] if labels else []

def performer_v2(slug, rig, ev, cues, env, place, character, base_look=None):
    prof = character.get("profile", {})
    return {"slug": slug, "rig_url": "/characters/%s/rig/rig.json" % slug, "place": place,
            "movement": float(prof.get("movement", 0.4)), "energy": float(prof.get("energy", 0.5)),
            "clips": ev["clips"], "expr": ev["expr"], "looks": ev["looks"], "blinks": ev["blinks"], "still": ev["still"],
            "camera": ev["camera"], "mouth": cues, "env": env, "env_fps": FPS, "base_look": base_look}

def cmd_perform2(a):
    slug = a.character
    character, delivery, gdir = load_graph(slug)
    if getattr(a, "set", None):
        delivery = load_set(a.set)
    character["_slug"] = slug
    rig = ensure_rig(slug)
    meta = rig_meta(slug)
    out = os.path.join(ROOT, "out", slug + (("__" + delivery["scenario_id"]) if delivery.get("scenario_id") else "")); os.makedirs(out, exist_ok=True)
    t_all = time.time()
    chosen, labels = select_and_synth(character, delivery["beats"], out, a.seconds, slack=0.3, trim=True)
    t_tts = time.time() - t_all
    wav = os.path.join(out, "performance.wav")
    timing, dur = assemble(chosen, wav)
    text = " ".join(c["beat"]["text"] for c in chosen)
    cues, t_rh = rhubarb(wav, text, os.path.join(out, "mouth.json"))
    ev = director2.direct(character, [c["beat"] for c in chosen], timing, seed=a.seed)
    tl = {"page": "stage2.html", "fps": FPS, "duration": round(dur, 3), "background": meta.get("stage", "/assets/stage/office.png"), "fadeIn": 0.25,
          "captions": captions(timing),
          "tags": [{"t0": 0.3, "t1": 3.0, "name": meta.get("display_name", character["name"].split(",")[0]), "sub": meta.get("tagline", character.get("species", "")), "color": meta.get("tag_color", "#7ed957")}],
          "labels": voice_labels(labels),
          "performers": [performer_v2(slug, rig, ev, cues, envelope(wav), place_for(rig, **meta.get("place", {})), character)],
          "meta": {"beats": [c["beat"]["id"] for c in chosen], "tts": labels, "stack": "oddrig/1",
                   "graph": gdir, "set": delivery.get("source"), "run_id": delivery.get("run_id")}}
    tlp = os.path.join(out, "timeline_v2.json"); json.dump(tl, open(tlp, "w"), indent=1)
    if a.stills is not None and len(a.stills):
        print(render.still(tlp, a.stills, os.path.join(out, "v2.jpg"))); return
    mp4 = os.path.join(out, "%s_oddrig_%ds.mp4" % (os.path.basename(out), int(round(dur))))
    st = render.render(tlp, wav, mp4, workers=a.workers)
    rep = {"mp4": mp4, "duration_s": round(dur, 2), "beats": tl["meta"]["beats"], "tts": labels, "tts_s": round(t_tts, 1), "rhubarb_s": t_rh,
           "clips": [(c["clip"], c["t0"], c.get("why")) for c in ev["clips"] if c.get("why") not in ("idle",) and not str(c.get("why", "")).startswith(("stress", "hand-beat"))],
           "actions": ev["log"], **st, "total_s": round(time.time() - t_all, 1)}
    json.dump(rep, open(os.path.join(out, "report_v2.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

def rig_meta(slug):
    """display name / tagline / voice fallback / framing: rig.v1 rig.json (legacy) or characters/<slug>/meta_rig.json"""
    m = {}
    for p in (os.path.join(ROOT, "characters", slug, "rig.json"), os.path.join(ROOT, "characters", slug, "stage.json")):
        if os.path.exists(p): m.update(json.load(open(p)))
    m.pop("place", None)
    sp = os.path.join(ROOT, "characters", slug, "stage.json")
    if os.path.exists(sp) and "place" in json.load(open(sp)): m["place"] = json.load(open(sp))["place"]
    return m

def cmd_rig(a):
    """Build (or rebuild) the OddRig for a character and render validation stills: bind pose + clip poses."""
    slug = a.character
    rig = ensure_rig(slug, rebuild=True)
    out = os.path.join(ROOT, "out", slug); os.makedirs(out, exist_ok=True)
    poses = [("rest", None, 0.6), ("suit_adjust", "suit_adjust", 0.75), ("shrug", "shrug", 0.55), ("point", "point_side_r", 0.75), ("explain", "gesture_talk_r", 0.9),
             ("arms_wide", "arms_wide", 0.9), ("look 3/4", "look_away", 0.6), ("glitch", "eye_glitch", 0.3)]
    clips = []; t = 0.0; stamps = []
    for label, clip, at in poses:
        if clip: clips.append({"clip": clip, "t0": t})
        stamps.append(round(t + at, 3)); t += 3.0
    character = load_graph(slug)[0] if True else {}
    tl = {"page": "stage2.html", "fps": FPS, "duration": t, "background": "/assets/stage/office.png", "captions": [], "tags": [],
          "performers": [{"slug": slug, "rig_url": "/characters/%s/rig/rig.json" % slug, "place": place_for(rig, **rig_meta(slug).get("place", {})),
                          "movement": 0.5, "energy": 0.5, "clips": clips + [{"clip": "idle_breathe", "t0": 0, "dur": t}], "expr": [{"t": 0, "expr": "neutral"}],
                          "looks": [], "blinks": [], "still": [], "camera": [], "mouth": [{"t": 0, "v": "X"}, {"t": stamps[4] - 0.2, "v": "D"}]}]}
    tlp = os.path.join(out, "rig_validate.json"); json.dump(tl, open(tlp, "w"))
    outs = render.still(tlp, stamps, os.path.join(out, "rigv.jpg"))
    from PIL import Image, ImageDraw
    ims = [Image.open(o).resize((360, 640)) for o in outs]
    sheet = Image.new("RGB", (360 * 4, 640 * 2 + 40), (20, 20, 28)); d = ImageDraw.Draw(sheet)
    for k, im in enumerate(ims):
        x, y = (k % 4) * 360, (k // 4) * 680
        sheet.paste(im, (x, y + 40)); d.text((x + 10, y + 12), poses[k][0], fill=(255, 255, 255))
    vp = os.path.join(out, "rig_validate.jpg"); sheet.save(vp, quality=88)
    print(json.dumps({"rig": rig_path(slug), "validate_sheet": vp, "bind_still": os.path.join(ROOT, "characters", slug, "rig", "validate.png"),
                      "bones": len(rig["skeleton"]["bones"]), "views": [v["view"] for v in rig["face"]["views"]], "hands": rig["hands"]["set"],
                      "landmarks": rig["provenance"]["report"]["landmarks"]}, indent=1))

def cmd_still(a):
    slug = a.character; out = os.path.join(ROOT, "out", slug)
    print(render.still(os.path.join(out, "timeline.json"), a.t, os.path.join(out, "still.jpg")))

def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("perform"); q.add_argument("--character", required=True); q.add_argument("--seconds", type=float, default=15)
    q.add_argument("--seed", type=int, default=11); q.add_argument("--workers", type=int, default=2); q.add_argument("--stills", type=float, nargs="*")
    q.add_argument("--legacy", action="store_true", help="old flat cut-out renderer (stage.js)")
    q.add_argument("--set", default=None, help="render a pipeline set JSON (chars freaks/<slug>/sets/*.json) instead of delivery.json")
    q = sp.add_parser("rig"); q.add_argument("--character", required=True)
    q = sp.add_parser("onboard"); q.add_argument("--slug", required=True); q.add_argument("--from-graph", default=None); q.add_argument("--prompt", default=None)
    q.add_argument("--photo", default=None); q.add_argument("--reuse-art", action="store_true")
    q = sp.add_parser("still"); q.add_argument("--character", required=True); q.add_argument("--t", type=float, nargs="+")
    q = sp.add_parser("add-character"); q.add_argument("--from-subject", required=True); q.add_argument("--photo", required=True); q.add_argument("--skip-art", action="store_true")
    q = sp.add_parser("duet"); q.add_argument("--script", required=True); q.add_argument("--workers", type=int, default=2); q.add_argument("--stills", type=float, nargs="*")
    q.add_argument("--legacy", action="store_true")
    a = p.parse_args()
    if a.cmd == "perform": (cmd_perform if a.legacy else cmd_perform2)(a)
    elif a.cmd == "rig": cmd_rig(a)
    elif a.cmd == "onboard":
        sys.path.insert(0, os.path.join(ROOT, "rigkit")); import onboard_rig; onboard_rig.run(a)
    elif a.cmd == "still": cmd_still(a)
    elif a.cmd == "add-character":
        import onboard; onboard.add_character(a)
    elif a.cmd == "duet":
        if a.legacy:
            import duet; duet.run(a)
        else:
            import duet2; duet2.run(a)

if __name__ == "__main__":
    main()
