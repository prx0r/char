"""Stage director v2: character graph + delivery beats + measured audio -> clip instances on an OddRig.

Same rules as director.py (expression per beat, graph-triggered actions rationed by gesture_frequency,
punchline stillness + camera push, blinks, eye-contact glances), but every movement is an authored clip
from clips/ resolved through clips/action_map.json, so any rig with oddrig-humanoid bones performs it.

Body language for talking:
  - idle loop always on, variant changes per beat (breathe / sway / still) with long crossfades
  - nods (head squash/stretch) on the stressed words
  - hand beats: a held talking gesture on some beats (budget from gesture_frequency), small beat
    flicks on stressed words while a gesture is held
"""
import json, os, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIPDIR = os.path.join(ROOT, "clips")
AMAP = json.load(open(os.path.join(CLIPDIR, "action_map.json")))
CLIPS = {n: json.load(open(os.path.join(CLIPDIR, n + ".json"))) for n in json.load(open(os.path.join(CLIPDIR, "index.json")))}

EXPR_MAP = {"deadpan": "deadpan", "neutral": "neutral", "annoyed": "annoyed", "surprised": "surprised", "confused": "confused",
            "smug": "smug", "grin": "grin", "happy": "happy", "warm": "happy", "excited": "surprised", "sad": "sad", "cheeky": "smug",
            "angry": "annoyed", "content": "self_satisfied", "self_satisfied": "self_satisfied", "confident": "confident"}


def resolve_clip(action):
    """character.json action -> (clip, mirror). Explicit `clip` wins, then tags, then body."""
    if not action: return None, False
    if action.get("clip"): return action["clip"], bool(action.get("mirror"))
    for tg in action.get("tags", []):
        if tg in AMAP["tags"]:
            return AMAP["tags"][tg], False
    return AMAP["body"].get(action.get("body")), False


def clip_dur(name):
    return CLIPS[name]["duration"] if name in CLIPS else 1.0


def direct(character, beats, timing, seed=11, t_offset=0.0, partner_side=None):
    rnd = random.Random(seed)
    prof = character.get("profile", {})
    gf = float(prof.get("gesture_frequency", 0.3)); eye_contact = float(prof.get("eye_contact", 0.85))
    energy = float(prof.get("energy", 0.5)); movement = float(prof.get("movement", 0.4))
    acts = character.get("actions") or {}
    ev = {"clips": [], "expr": [], "looks": [], "blinks": [], "still": [], "camera": [], "speaking": [], "log": []}
    budget = max(1, round(gf * len(beats) + 0.5)) + 1
    last_g = [-99.0]
    side_flip = [rnd.random() < 0.5]

    def add_clip(name, t0, dur=None, why="", w=1.0, mirror=False, force=False, gesture=True, **kw):
        if not name or name not in CLIPS: return False
        if gesture and not force:
            if budget_left[0] <= 0 or t0 - last_g[0] < 2.2: return False
        inst = {"clip": name, "t0": round(t0, 3), "why": why}
        if dur: inst["dur"] = round(dur, 3)
        if w != 1.0: inst["w"] = round(w, 3)
        if mirror: inst["mirror"] = True
        inst.update(kw)
        ev["clips"].append(inst)
        if gesture:
            budget_left[0] -= 1; last_g[0] = t0
        return True
    budget_left = [budget]
    talk_left = [1 + round(gf * len(beats) * 1.5 + energy)]

    def action_by(trigger=None, body=None):
        for name, a in acts.items():
            if trigger and a.get("trigger") == trigger: return name, a
            if body and a.get("body") == body: return name, a
        return None, None

    def fire_action(name, a, t0, hold_until=None, why="", force=False):
        clip, mir = resolve_clip(a)
        tags = a.get("tags", [])
        dur = None
        if clip and hold_until and CLIPS[clip].get("hold_at") is not None:
            dur = max(clip_dur(clip), hold_until - t0)
        ok = add_clip(clip, t0, dur, why="%s:%s" % (name, why), mirror=mir, force=force) if clip else False
        span = dur or (clip_dur(clip) if clip else 1.2)
        if (ok or not clip) and "gaze.away" in tags and clip not in ("look_away", "suit_adjust", "mime_swing"):
            ev["looks"].append({"t": round(t0 + 0.05, 3), "x": rnd.choice([-1, 1]) * 0.75, "y": 0.2, "why": name})
            ev["looks"].append({"t": round(t0 + span * 0.85, 3), "x": 0, "y": 0, "why": "back"})
        if not clip and a.get("body") == "hold still":
            ev["still"].append({"t0": round(t0, 3), "t1": round(t0 + 1.4, 3)})
        ev["log"].append((round(t0, 2), name, clip, "fired" if ok else "skipped"))
        return ok

    # idle: one loop instance per beat with long crossfades (three variants, never the same twice running)
    idle_cycle = ["idle_breathe", "idle_sway", "idle_still"]
    prev_idle = None
    t_start = t_offset
    for i, (b, tm) in enumerate(zip(beats, timing)):
        t0, t1, hold_end = tm["t0"] + t_offset, tm["t1"] + t_offset, tm["hold_end"] + t_offset
        perf = b.get("performance", {}); btype = b.get("type", "setup"); g = perf.get("gesture", "normal")
        if btype == "punchline" or g == "hold still": idle = "idle_still"
        else: idle = rnd.choice([x for x in idle_cycle if x != prev_idle])
        s0 = (t_start if i == 0 else t0 - 0.4)
        e0 = (hold_end + 0.6) if i + 1 < len(beats) else hold_end + 3
        ev["clips"].append({"clip": idle, "t0": round(s0 - (1.0 if i == 0 else 0), 3), "dur": round(e0 - s0 + 1.0, 3), "fade_in": 0.8 if i else 0.01, "fade_out": 0.8, "phase": round(rnd.uniform(0, 3), 2), "why": "idle", "w": 0.9})
        prev_idle = idle

        expr = EXPR_MAP.get(perf.get("expression", "neutral"), "neutral")
        ev["speaking"].append({"t0": round(t0, 3), "t1": round(t1, 3), "beat": b["id"]})
        ev["expr"].append({"t": round(max(0, t0 - 0.15), 3), "expr": expr, "beat": b["id"]})
        emph = float(b.get("speech", {}).get("emphasis", 0.5))
        words = [w for w in tm["words"] if len(w["w"].strip(".,?!")) > 3]
        stressed = sorted(words, key=lambda w: -(w["t1"] - w["t0"]))
        # nods on stressed words
        k = 0 if g == "hold still" else max(0, int(round(emph * energy * 3)))
        for w in stressed[:k]:
            ev["clips"].append({"clip": "nod", "t0": round(w["t0"] + t_offset - 0.2, 3), "amp": round(0.6 + emph * 0.7, 2), "why": "stress:" + w["w"]})
        # blinks / stillness
        if btype == "punchline" or g == "hold still":
            ev["still"].append({"t0": round(t0, 3), "t1": round(hold_end, 3)})
            ev["camera"].append({"t0": round(t0, 3), "t1": round(hold_end, 3), "kind": "push", "amt": 0.07})
            ev["blinks"].append({"t": round(hold_end - 0.35, 3), "slow": True})
        else:
            t = t0 + rnd.uniform(0.6, 1.4)
            while t < t1 - 0.3:
                ev["blinks"].append({"t": round(t, 3), "slow": False}); t += rnd.uniform(2.4, 4.6)
            if rnd.random() < (1 - eye_contact) * 2.5 and t1 - t0 > 2.0:
                tg = t0 + (t1 - t0) * rnd.uniform(0.3, 0.6)
                ev["looks"].append({"t": round(tg, 3), "x": rnd.choice([-0.6, 0.6]), "y": -0.35, "why": "think"})
                ev["looks"].append({"t": round(tg + 0.7, 3), "x": 0, "y": 0, "why": "back"})

        fired = False
        # graph-triggered character actions
        if btype == "punchline":
            name, a = action_by(trigger="after_punchline")
            if name:
                if t1 - t0 < 1.6: fired = fire_action(name, a, t0 - 0.35, hold_end, "after_punchline(reveal)", force=True)
                else: fired = fire_action(name, a, t1 + 0.12, hold_end, "after_punchline", force=True)
        if btype == "closer" and g != "hold still":
            name, a = action_by(trigger="before_closer")
            if name: fired = fire_action(name, a, t0 - 0.35, hold_end + 0.3, "before_closer", force=True)
        if not fired and (expr == "annoyed" or g == "small gesture"):
            name, a = action_by(trigger="on_groan")
            if name: fired = fire_action(name, a, t0 + min(1.0, (t1 - t0) * 0.35), None, "on_groan")
        if not fired and btype == "setup" and i == 0:
            name, a = action_by(trigger="on_enter")
            if name: fired = fire_action(name, a, t0, None, "on_enter")
        if not fired:
            dg = AMAP["delivery_gesture"].get(g)
            if g == "lean in":
                name, a = action_by(body="lean in")
                if name: fired = fire_action(name, a, t0 - 0.2, hold_end, "delivery:lean in")
            if not fired and dg:
                mir = False
                if dg == "look_away": mir = True   # 'look left' = screen-left for the audience
                if dg == "arms_wide" and movement < 0.45: dg = "present_palm_r"
                fired = add_clip(dg, t0 + (t1 - t0) * (0.35 if dg != "lean_in" else 0) - 0.2,
                                 max(clip_dur(dg), hold_end - t0) if CLIPS[dg].get("hold_at") is not None and dg in ("lean_in",) else None,
                                 why="delivery:" + g, mirror=mir)
        # talking gesture: hold a hand up through part of the beat, beat flicks on stressed words
        if not fired and g in ("normal", "big gesture") and btype != "punchline" and t1 - t0 > 1.8 and stressed:
            if talk_left[0] > 0 and rnd.random() < 0.45 + gf:
                side_flip[0] = not side_flip[0]
                ws = sorted(stressed[:3], key=lambda w: w["t0"])
                gt0 = ws[0]["t0"] + t_offset - 0.45
                dur = min(t1 - gt0 + 0.1, max(2.4, (ws[-1]["t1"] + t_offset) - gt0 + 0.9))
                name = "point_up_r" if (emph > 0.75 and energy > 0.45 and rnd.random() < 0.5) else "gesture_talk_r"
                if gt0 - last_g[0] > 1.6 and add_clip(name, gt0, dur, why="talk-gesture", mirror=side_flip[0], force=True):
                    talk_left[0] -= 1; last_g[0] = gt0
                    hold_from, hold_to = gt0 + CLIPS[name].get("hold_at", 1.0) * 0.6, gt0 + dur - 0.7
                    for w in ws:
                        tw = w["t0"] + t_offset - 0.22
                        if hold_from < tw < hold_to:
                            ev["clips"].append({"clip": "beat_r", "t0": round(tw, 3), "mirror": side_flip[0], "amp": round(0.7 + 0.6 * emph, 2), "why": "hand-beat:" + w["w"]})
        # ending: look back to audience
    ev["looks"].sort(key=lambda x: x["t"]); ev["blinks"].sort(key=lambda x: x["t"])
    bl = []
    for b in ev["blinks"]:
        if not bl or b["t"] - bl[-1]["t"] > 1.2: bl.append(b)
    ev["blinks"] = bl
    ev["clips"].sort(key=lambda c: c["t0"])
    ev["profile"] = {"energy": energy, "movement": movement}
    return ev
