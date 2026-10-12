# Rigging stack: OddRig (oddrig/1)

Constraints: an agent with no GUI editor, on an aarch64 sandbox (2 vCPU, ~8 GB RAM, no GPU), using no paid
APIs. Output must be deterministic 1080×1920 MP4s, and characters come from graphs (`character.json` +
`delivery.json`).

## Decision

**OddRig is our own open, JSON, code-authored skeletal format** (the semantics are borrowed from
Spine/DragonBones: bones, slots, skinned meshes, attachments). It is played by a **small WebGL2 runtime in
headless Chromium** and fed by a **Python onboarding pipeline** that turns part-sheet art into a rig with no
hand-placed points. All characters share **one clip library on standard humanoid bones**.

```
graph (character.json, delivery.json)            look prompt / likeness ref
        │                                                │
        │                     rigkit/onboard_rig.py ─────┤ writes the 3 house-style sheet prompts
        │                                                ▼
        │                     image model (Hark generate_image) → heads / parts / hands sheets (magenta bg)
        │                                                │
        │                     rigkit/segment.py   chroma key, un-mix, despill, components
        │                     rigkit/features.py  silhouette joints; MediaPipe face landmarks (human heads)
        │                                         or cartoon eye/mouth detector; cross-view checks
        │                     rigkit/build_rig.py grid meshes + weights, head views + depth, hands + grip,
        │                                         anchors, physics → characters/<slug>/rig/rig.json + tex/
        │                     rigkit/validate.py  JSON-Schema + "every shared clip resolves on this rig"
        ▼                                                │
pipeline/director2.py  beats + measured audio → clip instances (clips/*.json via clips/action_map.json),
                       expressions, gaze, blinks, stillness, camera
        │
renderer/stage2.html   oddrig.js (skinning, head views, face) + stage2.js (clip mixer, IK, springs, stage)
        │              frame-stepped in Playwright/Chromium, JPEG → x264, AAC mux (pipeline/render.py)
        ▼
out/<slug>/<slug>_oddrig_<N>s.mp4
```

### Why this wins for this team

- **An agent can author it.** Rigs, clips and overrides are plain JSON that diffs cleanly. None of the
  candidate editors (Spine, Rive, Inochi Creator, DragonBones Pro) can be driven headless here, and every
  one of their binary or editor-owned formats would make the agent depend on a GUI round-trip.
- **Reuse is structural.** It has three parts:
  - Standard bone names (`oddrig-humanoid/1`).
  - Translations measured in `unit`, which is shoulder width.
  - IK targets given as named anchors (`tie_knot`, `chest_center`, `belly`, …).

  Together these mean one `suit_adjust` clip plays on Ralph (a pinch at his tie), on Zeus (his hand goes to
  his beard and chin, because his `tie_knot` anchor falls back to the collar line) and on Chris, with no
  character code. Per-character differences live in data: `character.json` actions (an explicit `clip`
  wins) and `rig.json clips.overrides`.
- **Quality where it shows.**
  - *Smooth bends.* When a vertex is weighted to a parent/child pair (shoulder→elbow, neck→head, tie chain),
    the mesh uses rotational-blend skinning, so elbows bend without seams or a "candy-wrapper" collapse.
  - *Head turns.* Each character has multi-angle head art (front, 3/4, profile, mirrored to the other side).
    The runtime crossfades views in a narrow yaw band, and per-vertex depth adds parallax inside a view.
  - *Faces stay on-model.* Faces are drawn into the painted head art, clipped to the painted eye mask.
  - *Live motion.* Hands are swappable drawings. Secondary motion comes from deterministic springs (tie,
    wrist drag, head lag).
- **Deterministic and cheap.** Clip mixing, IK and physics are precomputed per frame in one sequential pass, so
  frame N is identical regardless of which worker renders it. Measured render speed is about 4.6 fps at
  1080×1920 with 2 workers.
- **Exportable later.** The rig is Spine/DragonBones-shaped, so a JSON exporter to DragonBones (open format;
  its Pixi v8 runtime loads here) or Spine-JSON is a mapping exercise. Rive's web runtime also runs in this
  Chromium (verified), so a `.riv` could be a playback target if someone authors it in the Rive editor.

## Candidates evaluated (≤2 failed loops each)

| Candidate | Licence | What was actually run here | Verdict / blocker |
|---|---|---|---|
| **Spine runtimes** | Spine Runtimes License: each user of a product that integrates them "must obtain their own Spine Editor license" (paid editor). Text fetched from `EsotericSoftware/spine-runtimes/4.2/LICENSE`. | Nothing (licence check only) | **Rejected**: licence-gated on the paid editor, and the editor is GUI-only. |
| **DragonBones** (open JSON format) | Runtime port `dragonbones-pixijs@1.0.5` (ISC, Pixi v8, published 2025-05). The original Egret editor is abandoned (per that package's README). | `npm install`; the ES module imports in headless Chromium alongside Pixi 8.22 (exports `Armature`, `Animation`, …) | **Not chosen.** The format is viable, but no headless authoring exists. We would be emitting its JSON ourselves, and OddRig already is that JSON with our face/view extensions. Kept as a possible export target. No armature was rendered. |
| **PixiJS** + mesh/rope + custom skeleton | MIT | Pixi 8.22.0 initialised with the WebGL renderer in headless Chromium and drew a `MeshRope` from Ralph's sleeve texture (`vendors/eval/pixi_frame.png`) | **Viable, not needed.** Pixi gives meshes but no skinning, IK or face system. Our 600-line runtime does exactly what we need, with zero dependencies. Could replace `GLStage` later if we want filters or batching. |
| **Inochi2D** (open Live2D-like) | BSD-2-Clause (Inochi Creator, inox2d) | Read only: no `cargo`/`dub`/`ldc2` on this box | **Rejected for authoring.** Rigging happens in Inochi Creator, a GUI app (D + imgui). The inox2d README says it is "in a prototype state, not recommended for production", with physics and animations not yet supported. Its deformer ideas (mesh warps, parameters) are mirrored here as the brow-region warp and depth parallax. |
| **Meta Animated Drawings** | MIT | Cloned `facebookresearch/AnimatedDrawings@b859684`; venv; `pip install --no-deps -e .`; `PYOPENGL_PLATFORM=osmesa` render of the `dab` example | **Blocked after 2 loops.** (1) `ModuleNotFoundError: pkg_resources`, fixed with `setuptools<70`. (2) `AttributeError: module 'numpy' has no attribute 'bool8'`: it pins numpy 1.24.4 / scipy 1.10, which have no wheels for this Python 3.12, and its detector needs TorchServe (mmdet/mmpose). Even when working, it ARAP-deforms a *single* flat drawing: an arm in front of the torso smears, and there are no hidden parts or head turns. |
| **Rive runtimes** | Runtime MIT; editor/CLI separate (CLI has no aarch64 build: see CAPABILITY_REPORT) | `@rive-app/canvas@2.44.2` in headless Chromium, wasm served locally (`RuntimeLoader.setWasmUrl`), loaded `cdn.rive.app/animations/vehicles.riv` → artboards `Truck, Jeep`, state machine `bumpy`, rendered (`vendors/eval/rive_frame.png`) | **The runtime works here; authoring is blocked.** `.riv` files come from the Rive editor (GUI) or the CLI (x86_64 only). Not usable as the agent's authoring format; possible future playback target. |
| **MediaPipe** (auto-rig landmarks) | Apache-2.0 | `mediapipe 1.1.0` on aarch64. Needed `libegl1` (`OSError: libEGL.so.1` before that). Pose Landmarker found Chris's shoulders/elbows/hips on his sheet but nothing on Ralph. Face Landmarker found **no** face on the full cartoon sheets, but **does** fire on the cropped head views: Chris front/3/4/profile, Zeus front | **Adopted for human heads** (eye contours → eye masks, inner lips → mouth line, surround colour for bearded mouths). Lizards and other non-human faces use the cartoon detector. |

## Rig schema (rigkit/rig.schema.json, `oddrig/1`)

- **skeleton**: bind pose = world positions, zero rotation. Required bones: `root hips spine chest neck head
  shoulder_{r,l} upper_arm_{r,l} lower_arm_{r,l} hand_{r,l}`. Optional secondary chains (`tie_0..`) are
  marked `secondary`. `_r` is the character's right (screen-left in a front view).
- **slots**: draw order, with slot kinds `head_views` and `hand`.
- **attachments**: meshes, each with:
  - texture, bind vertices, uvs, triangles;
  - per-vertex weights (≤4; parent/child pairs use rotational blend, everything else LBS);
  - `depth` on head views (for parallax);
  - `grip` on hands (the point IK aims at the target).
- **hands**: standard set `rest palm point fist pinch flat`, as per-side attachment maps. Clips switch them
  with step keys.
- **face**:
  - `views`: yaw −1…1, real or `mirror_of`. Each view has painted-eye ellipses, a mouth line
    `y = ax² + bx + c`, a hinge side for the jaw, skin/ink colours and the detector used.
  - `eyes`: `ball` or `human`.
  - `mouth`: `jaw` or `vector`.
  - `visemes`: Rhubarb A–H + X.
  - `expressions`: neutral, deadpan, annoyed, surprised, confused, smug, grin, self_satisfied, confident,
    suspicious, embarrassed, sad.
- **anchors**: named IK targets bound to bones.
- **physics**: springs (stiffness, damping, gain).
- **limits**, **clips** (`inherit: shared`, `overrides`) and **provenance** (sheets plus a landmark report).

Run `python3 rigkit/validate.py <slug>...` to check the schema and confirm every shared clip target exists
on the rig. All three rigs pass with 24 clips.

## Clip library (clips/, `oddclip/1`)

Keys are `[t, value, ease]`; the ease is one of `l i o io b s`. Track names are `bone.channel`, with `*`
meaning both sides. Other track types are `face.*`, `ik.hand_x.{w,ox,oy}`, `hand_x.pose` and `fx.glitch`.

Sign conventions:
- Side bones: + is outward, the same anatomical meaning on both sides.
- Centre bones: + is clockwise.
- Translations are in rig units.

Clip settings:
- `hold_at` stretches a clip by holding its pose (used by leans and held gestures).
- `mirror` swaps sides.
- `base` composes clips (`warm_rock_lean` = `lean_in` + basking face).

Clips by group:

| Group | Clips |
|---|---|
| Base / idle | `rest_pose`, `idle_breathe`, `idle_sway` (weight shift), `idle_still` |
| Talk | `nod` (squash/stretch), `beat_r`, `gesture_talk_r`, `point_up_r`, `fist_beat_r` |
| Gesture | `point_side_r`, `present_palm_r`, `shrug`, `lean_in`, `warm_rock_lean`, `arms_wide`, `head_tilt`, `suit_adjust` (IK), `mime_swing` (two-hand IK) |
| Look | `look_away`, `look_up`, `look_audience` (eyes lead; the head follows on a spring) |
| React | `eye_glitch`, `react_recoil`, `laugh` |

`clips/action_map.json` resolves graph actions to clips: an explicit `clip` wins, then the 40-odd `tags`
used across freaks (`gesture.adjust`, `pose.bask`, `reaction.glitch`, `gesture.summon`, …), then `body`.
`director2.py` fires these by the graph triggers `after_punchline`, `before_closer`, `on_groan`,
`on_emphasis` and `on_enter`, and maps the delivery gestures `shrug`, `lean in`, `big gesture` and
`look left`.

## Onboarding (run.py onboard / rig)

1. `run.py onboard --slug S [--from-graph G] [--prompt "..."] [--photo likeness.png]` writes
   `characters/S/onboard.json` and `art/sheet_prompts.json`. These hold three prompts in house style
   (style lock = Ralph's sheet) plus the reference images.
2. Generate the three sheets with the image model, saved to `assets/gen/parts/S_{heads,parts,hands}.png`.
   This is a tool call, not Python.
3. Run `run.py onboard` again, or `run.py rig --character S`. This segments the sheets, builds the rig and
   writes two validation outputs:
   - `characters/S/rig/validate.png`: bind pose with joints and features;
   - `out/S/rig_validate.jpg`: the rig posed by eight shared clips.

Joints come from part silhouettes: the shoulder line from the torso width profile, the collar hole or neck
notch, arm caps and cuffs, wrists, and the chin row. No point is hand-placed.

## Known limits (honest)

- **No real 3D turn.** Turns are view swaps plus parallax. Mid-turn frames crossfade two drawings for about
  0.1 yaw.
- **Fixed parts sheet.** The image model must follow the parts-sheet layout. Zeus's sheet came back with
  extra loose pieces, which the classifier ignores. A failed sheet means regenerating it.
- **Feature detection is heuristic for non-humans.** Zeus's 3/4 and profile views and Ralph's profile have
  no animated eyes. A false eye was dropped by a cross-view check, but the painted eyes stay static in
  those views.
- **No props.** The golf mime has no club. There are no prop slots yet.
- **Single-plane arms.** Arms can't reach toward the camera; it's 2D.
