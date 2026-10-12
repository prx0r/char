"""Stage director: character graph + delivery beats + measured audio -> timed performance events.

Principles (from the brief, implemented as rules, not per-line gestures):
- expression per beat, eased in ~150 ms before the words start
- character actions fire on their graph triggers (after_punchline, on_groan, before_closer),
  rationed by profile.gesture_frequency with a minimum spacing
- punchlines: stillness (head motion damped), camera push, hold for profile.punchline_hold_ms
- blinks: seeded 2.5-5 s rhythm, a blink at beat changes, none during a punchline stare
  except one slow blink as the hold releases
- look: audience by default; profile.eye_contact controls glance-aways; gaze.* tags of
  actions override
"""
import random, math

EXPR_MAP = {  # delivery expression -> rig expression
    "deadpan": "deadpan", "neutral": "neutral", "annoyed": "annoyed", "surprised": "surprised",
    "confused": "confused", "smug": "smug", "happy": "happy", "warm": "happy", "excited": "surprised",
    "sad": "sad", "cheeky": "smug", "angry": "annoyed", "content": "content",
}

def _action_for(character, trigger=None, body=None):
    for name, a in (character.get("actions") or {}).items():
        if trigger and a.get("trigger") == trigger: return name, a
        if body and a.get("body") == body: return name, a
    return None, None

def direct(character, beats, timing, seed=11, speaker_id="A", t_offset=0.0):
    """beats: delivery beats (selected). timing: list per beat {t0,t1,words,hold_end}.
    Returns dict of tracks for one performer."""
    rnd = random.Random(seed)
    prof = character.get("profile", {})
    gf = float(prof.get("gesture_frequency", 0.3))
    eye_contact = float(prof.get("eye_contact", 0.85))
    hold = prof.get("punchline_hold_ms", 1000) / 1000.0
    energy = float(prof.get("energy", 0.5))
    movement = float(prof.get("movement", 0.4))
    ev = {"expr": [], "gestures": [], "looks": [], "blinks": [], "nods": [], "still": [], "camera": [], "speaking": []}
    budget = max(1, round(gf * len(beats) + 0.5)) + 1   # actions available in this clip
    last_g = -99.0
    min_gap = 2.5
    acts = character.get("actions") or {}

    def add_gesture(name, t0, dur, tags=(), why=""):
        nonlocal budget, last_g
        if budget <= 0 or t0 - last_g < min_gap: return False
        ev["gestures"].append({"name": name, "t0": round(t0, 3), "t1": round(t0 + dur, 3), "why": why})
        budget -= 1; last_g = t0
        for tg in tags:
            if tg == "gaze.away":
                ev["looks"].append({"t": round(t0 + 0.05, 3), "x": rnd.choice([-1, 1]) * 0.8, "y": 0.25, "why": name})
                ev["looks"].append({"t": round(t0 + dur * 0.85, 3), "x": 0, "y": 0, "why": "back"})
            if tg == "gaze.audience":
                ev["looks"].append({"t": round(t0, 3), "x": 0, "y": 0, "why": name})
        return True

    n = len(beats)
    for i, (b, tm) in enumerate(zip(beats, timing)):
        perf = b.get("performance", {})
        btype = b.get("type", "setup")
        expr = EXPR_MAP.get(perf.get("expression", "neutral"), "neutral")
        t0, t1, hold_end = tm["t0"] + t_offset, tm["t1"] + t_offset, tm["hold_end"] + t_offset
        ev["speaking"].append({"t0": round(t0, 3), "t1": round(t1, 3), "beat": b["id"]})
        ev["expr"].append({"t": round(max(0, t0 - 0.15), 3), "expr": expr, "beat": b["id"]})
        # emphasis nods on stressed words: longest content words, scaled by beat emphasis & energy
        emph = float(b.get("speech", {}).get("emphasis", 0.5))
        words = [w for w in tm["words"] if len(w["w"].strip(".,?!")) > 3]
        k = 0 if perf.get("gesture") == "hold still" else max(0, int(round(emph * energy * 3)))
        for w in sorted(words, key=lambda w: -(w["t1"] - w["t0"]))[:k]:
            ev["nods"].append({"t": round(w["t0"] + t_offset, 3), "amp": round(0.5 + emph * 0.8, 2)})
        # blinks inside the beat
        if btype == "punchline" or perf.get("gesture") == "hold still":
            ev["still"].append({"t0": round(t0, 3), "t1": round(hold_end, 3)})
            ev["camera"].append({"t0": round(t0, 3), "t1": round(hold_end, 3), "kind": "push", "amt": 0.07})
            ev["blinks"].append({"t": round(hold_end - 0.35, 3), "slow": True})
        else:
            t = t0 + rnd.uniform(0.6, 1.4)
            while t < t1 - 0.3:
                ev["blinks"].append({"t": round(t, 3), "slow": False})
                t += rnd.uniform(2.4, 4.6)
            # occasional thinking glance (1 - eye_contact)
            if rnd.random() < (1 - eye_contact) * 2.5 and t1 - t0 > 2.0:
                tg = t0 + (t1 - t0) * rnd.uniform(0.3, 0.6)
                ev["looks"].append({"t": round(tg, 3), "x": rnd.choice([-0.6, 0.6]), "y": -0.35, "why": "think"})
                ev["looks"].append({"t": round(tg + 0.7, 3), "x": 0, "y": 0, "why": "back"})
        # character actions by graph trigger
        g = perf.get("gesture", "normal")
        if btype == "punchline":
            name, a = _action_for(character, trigger="after_punchline")
            if name:
                if t1 - t0 < 1.6:   # short reveal punchline ("See? Slit."): react in the gap, hold through the line
                    s0 = t0 - 0.35; add_gesture(name, s0, hold_end - s0 - 0.1, a.get("tags", []), "after_punchline(reveal)")
                else:
                    add_gesture(name, t1 + 0.12, min(1.4, max(0.9, hold_end - t1)), a.get("tags", []), "after_punchline")
        if btype == "closer" or (i + 1 < n and beats[i + 1].get("type") == "closer" and False):
            name, a = _action_for(character, trigger="before_closer")
            if name:
                budget += 1; last_g = -99  # structural: the closer action always plays
                add_gesture(name, t0 - 0.25, (hold_end - t0) + 0.25, a.get("tags", []), "before_closer")
        if expr == "annoyed" or g == "small gesture":
            name, a = _action_for(character, trigger="on_groan")
            if name: add_gesture(name, t0 + min(1.0, (t1 - t0) * 0.35), 1.5, a.get("tags", []), "on_groan")
        if g == "shrug":
            add_gesture("shrug", t0 + (t1 - t0) * 0.5, 0.9, [], "delivery:shrug")
        if g == "lean in" and not any(x["t0"] >= t0 - 0.3 for x in ev["gestures"]):
            name, a = _action_for(character, body="lean in")
            add_gesture(name or "lean_in", t0 - 0.2, hold_end - t0 + 0.2, (a or {}).get("tags", []), "delivery:lean in")
    ev["looks"].sort(key=lambda x: x["t"]); ev["blinks"].sort(key=lambda x: x["t"])
    # keep blinks apart >= 1.2 s
    bl = []
    for b in ev["blinks"]:
        if not bl or b["t"] - bl[-1]["t"] > 1.2: bl.append(b)
    ev["blinks"] = bl
    ev["profile"] = {"energy": energy, "movement": movement}
    return ev
