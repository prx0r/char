"""Character onboarding onto the OddRig stack.

  python3 run.py onboard --slug <slug> [--from-graph <chars-repo slug>] [--prompt "<look>"] [--photo <likeness ref>]

Stage A (always): writes characters/<slug>/onboard.json (rig build config) and art/sheet_prompts.json — the
three part-sheet prompts in house style (heads turnaround, body parts, hand poses), each with its reference
images (style lock = Ralph's sheet; likeness = --photo or an existing sheet). The look comes from --prompt,
else from the graph (species / persona / premise).
Stage B (when the three sheets exist at assets/gen/parts/<slug>_{heads,parts,hands}.png): segmentation ->
silhouette joints + face features -> rig.json -> validation stills (run.py rig).

Image generation itself is a tool call outside Python (Hark `generate_image`, an image model); Stage A prints
exactly what to run. Nothing in the rig is hand-placed.
"""
import json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHARS_REPO = os.environ.get("CHARS_REPO", "/workspace/work/chars")
STYLE_LOCK = "characters/reptilian-ralph-middle-management/art/base.png"
STYLE = ("EXACTLY the same art style as the style reference image (clean dark-brown outlines, flat cel shading, one soft "
         "shadow tone, premium adult TV-animation look)")
BG = "Plain solid flat magenta #FF00FF background. No text, no labels."

HANDS_LAYOUT = [{"pose": "rest", "wrist": "top"}, {"pose": "palm", "wrist": "bottom"}, {"pose": "point", "wrist": "bottom"},
                {"pose": "fist", "wrist": "bottom"}, {"pose": "pinch", "wrist": "bottom"}, {"pose": "flat", "wrist": "right"}]


def prompts(look, head_desc, torso_desc, arm_desc, hand_desc, likeness=False):
    lk = " Keep the face and likeness of the person in the likeness reference image, kind and respectful, not caricatured." if likeness else ""
    return {
        "heads": (f"Character model sheet of {look}. {STYLE}.{lk} Draw the HEAD AND NECK ONLY ({head_desc}), cut off cleanly at the "
                  "base of the neck (no collar, no clothing, no shoulders), three times, stacked in one vertical column, equal size, equal "
                  "spacing, each completely separated by plain background: TOP = front view facing the camera; MIDDLE = three-quarter view, "
                  "head turned toward the viewer's left; BOTTOM = full side profile facing the viewer's left. Neutral calm expression, mouth "
                  f"closed, eyes open. {BG} No body, no shadows on the background."),
        "parts": (f"Cut-out animation PARTS SHEET for {look}. {STYLE}.{lk} {BG} Every part is separated from every other part by clear "
                  "magenta gaps; nothing overlaps. TOP CENTER: the head and neck, front view. CENTER: the torso, front view, from the base of "
                  f"the neck down to the upper thighs: {torso_desc}. The torso is drawn WITHOUT ARMS: the shoulders end in smooth rounded "
                  f"seams, no sleeves at all. LEFT EDGE and RIGHT EDGE: the two arms, {arm_desc}, each hanging perfectly straight and "
                  "vertical from the rounded shoulder top down to the wrist, with NO hand. Below each arm: one relaxed hand. BOTTOM CENTER: "
                  "the hips and upper legs (trousers / lower garment) from the waist down, front view. Flat even lighting."),
        "hands": (f"HAND POSE SHEET for {look} ({hand_desc}). {STYLE}. Six separate drawings of the character's RIGHT hand, each one "
                  "starting at the wrist (wrist cut straight across, no sleeve), all the same size, arranged in a 2-column by 3-row grid "
                  "with wide magenta gaps, nothing overlapping: 1) relaxed hand hanging down, fingers loosely together pointing down; "
                  "2) open palm facing the viewer, fingers up and slightly spread; 3) pointing: index finger extended straight up, other "
                  "fingers curled; 4) closed fist, knuckles to the viewer; 5) pinch: thumb and index finger pinched together as if gripping "
                  f"a tie knot, other fingers curled; 6) relaxed hand palm-down, fingers pointing to the left, gesturing. {BG} No arms."),
    }


def run(a):
    slug = a.slug
    cdir = os.path.join(ROOT, "characters", slug); os.makedirs(os.path.join(cdir, "art"), exist_ok=True)
    graph = None
    gslug = a.from_graph or slug
    for base in (cdir, os.path.join(CHARS_REPO, "freaks", gslug)):
        p = os.path.join(base, "character.json")
        if os.path.exists(p): graph = json.load(open(p)); break
    spec_p = os.path.join(cdir, "art", "look.json")
    spec = json.load(open(spec_p)) if os.path.exists(spec_p) else {}
    if a.prompt: spec["look"] = a.prompt
    if not spec.get("look"):
        if not graph: raise SystemExit("need --prompt or a graph with species/persona")
        spec["look"] = "%s (%s): %s" % (graph.get("name"), graph.get("species", ""), (graph.get("persona") or {}).get("deal", graph.get("premise", "")))
    for k, dflt in (("head", "the character's head"), ("torso", "the character's top garment"), ("arms", "in the character's sleeves"), ("hands", "the character's hands")):
        spec.setdefault(k, dflt)
    json.dump(spec, open(spec_p, "w"), indent=1)
    refs = [STYLE_LOCK]
    if a.photo: refs.append(a.photo)
    P = prompts(spec["look"], spec["head"], spec["torso"], spec["arms"], spec["hands"], likeness=bool(a.photo))
    sheets = {k: "assets/gen/parts/%s_%s.png" % (slug, k) for k in ("heads", "parts", "hands")}
    json.dump({"style_lock": STYLE_LOCK, "refs": refs, "sheets": sheets, "prompts": P,
               "generator": "Hark generate_image (image model); one call per sheet with image_paths=refs"},
              open(os.path.join(cdir, "art", "sheet_prompts.json"), "w"), indent=1)
    ob_p = os.path.join(cdir, "onboard.json")
    ob = json.load(open(ob_p)) if os.path.exists(ob_p) else {}
    ob.setdefault("slug", slug); ob.setdefault("style", "pogtown-2d-v1"); ob["sheets"] = sheets
    ob.setdefault("head_views", ["front", "q_left", "profile_left"]); ob.setdefault("view_yaw", {"front": 0.0, "q_left": -0.5, "profile_left": -1.0})
    ob.setdefault("hands_layout", HANDS_LAYOUT); ob.setdefault("collar_hole", True); ob.setdefault("secondary", [])
    ob.setdefault("eyes", {"type": spec.get("eye_type", "human")}); ob.setdefault("mouth", {"type": spec.get("mouth_type", "vector"), "max_open": 0.07})
    json.dump(ob, open(ob_p, "w"), indent=1)
    missing = [k for k, p in sheets.items() if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        print(json.dumps({"stage": "A", "status": "needs_art", "missing": missing, "refs": refs,
                          "prompts_file": os.path.join(cdir, "art", "sheet_prompts.json")}, indent=1))
        return
    # Stage B
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "rig", "--character", slug], capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-1500:]) if r.returncode else None
