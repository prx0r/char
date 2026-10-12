# char: character graph → animated 2D performance (local experiment)

Reads Freaktown/Pogtown character graphs (`/workspace/work/chars/freaks/<slug>/character.json` + `delivery.json`)
and produces a 1080×1920 MP4 with synced audio.

```
./setup.sh                                                     # re-provision pip/apt (sandbox loses them)
python3 run.py perform --character reptilian-ralph-middle-management   # ~15-17 s solo
python3 run.py add-character --from-subject chris --photo auto --skip-art   # Chris rig + graph + voice
python3 run.py perform --character chris-prior-golf-dad
python3 run.py duet --script scripts/ralph_chris_duet.json             # two-hander, shared stage
python3 run.py perform --character <slug> --stills 1.0 4.2            # inspect frames only
```

Pipeline: delivery beats → beat selection to fit ~15 s (ends on the closer; pulls in a beat that
fires an unused character action) → TTS per beat (trim, measure) → assembled WAV with pauses and
punchline holds → Rhubarb mouth cues (A–H, X) + RMS envelope → `pipeline/director.py` events
(expressions, graph-triggered actions rationed by `gesture_frequency`, stillness and camera push on
punchlines, blinks, glances, emphasis nods) → `renderer/stage.js` in headless Chromium, frame-stepped →
x264 + AAC.

Layout
```
run.py                 CLI (perform / add-character / duet / still)
pipeline/director.py   stage director (graph + beats + timing → events)
pipeline/tts.py        kokoro (used), edge-tts, voxcpm adapters
pipeline/vox_worker.py VoxCPM2 voice-design worker (.venv-vox)
pipeline/prep_rig.py   cut generated sheet → head / jaw / body / collar / prop layers
pipeline/onboard.py    add-character (subject → rig.json, character.json, delivery.json, meta.json)
pipeline/duet.py       two-hander staging (listening behaviour, camera cuts)
pipeline/render.py     Playwright frame renderer + ffmpeg mux
renderer/stage.{html,js}  the rig: painted layers + procedural eyes/lids/brows/mouth, gestures, camera, captions
characters/<slug>/     rig.json, art/, layers/ (+ character.json, delivery.json for onboarded characters)
assets/                gen/ (generated art), stage/, fonts/
scripts/               duet scripts
experiments/           MANIFEST.md (vendors, SHAs, licences), CAPABILITY_REPORT.md
out/                   renders (gitignored)
```
Adding a Pogtown character: drop `characters/<slug>/rig.json` and art (a sheet in the style of `characters/reptilian-ralph-middle-management/art/base.png`).
The graph and delivery are read from the chars repo.
