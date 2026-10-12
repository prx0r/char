# Vendors & assets manifest

Vendors are shallow clones in `vendors/` (gitignored). Re-clone with the SHAs below.

| Vendor | Repo | Pinned SHA | Licence | Used? |
|---|---|---|---|---|
| Rhubarb Lip Sync 1.14.0 | github.com/DanielSWolf/rhubarb-lip-sync | 9b9573cd21b253c9ba58739bbd1aa0b50b991bff | MIT (deps MIT/BSD) | **Yes**: mouth cues. Built from source on aarch64: `cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --target rhubarb -j2` (needs `cmake build-essential libboost-all-dev`; the `all` target fails only on the Java "Rhubarb for Spine" extra) |
| VoxCPM2 | github.com/OpenBMB/VoxCPM | cce58a5b59303c9bd63d12f23afe6a49a4a80c59 | Apache-2.0 (code); weights openbmb/VoxCPM2 on HF | Wired in (`pipeline/vox_worker.py`, `.venv-vox/`), but **blocked** here: OOM-killed |
| Agent Theater | github.com/needle-tools/agent-theater | a4aabcb8c01318fcd46b1c7a339acada7ef185ca | MIT | Read only; not used |
| OpenMontage | github.com/calesthio/OpenMontage | 9327439db69021ab4b0e2776729bf3b58fdb5a87 | AGPL-3.0 (eval only, no code copied) | Read only; not used |
| rive-docs (cli/ only, sparse) | github.com/rive-app/rive-docs | e9d9681b9dc1d03d826f937adc5ec8fadf480677 | (docs) | Read: the CLI ships no aarch64 Linux build |

Runtime deps that are not vendored:
- Kokoro-82M ONNX (`kokoro-onnx` pip, Apache-2.0). Model files are in `.cache/kokoro/` (kokoro-v1.0.onnx, voices-v1.0.bin), from github.com/thewh1teagle/kokoro-onnx release `model-files-v1.0`. This is the TTS actually used for every clip.
- Playwright (Python, Apache-2.0) driving the **Debian `chromium` package**. Playwright's own arm64 Chromium download returns 403.
- ffmpeg (Debian).
- Fonts in `assets/fonts/`: Lilita One and Nunito (both SIL OFL, from google/fonts).

Generated art (image model via Hark `generate_image`; original, in one locked style):
- `assets/gen/ralph_base.png`: first Ralph design (landscape)
- `assets/gen/ralph_tall.png`: Ralph 9:16 sheet. This is the **style lock** for every other character.
- `assets/gen/ralph_blank.png`: blank-face edit (not used: Ralph keeps his painted eyes and mouth line, and the rig overdraws them)
- `assets/gen/ralph_hand.png`: hand prop for suit_adjust
- `assets/gen/chris_tall.png`: Chris sheet. Refs: ralph_tall.png (style) and chris/card/photo.jpg (likeness).
- `assets/gen/chris_blank.png`: Chris with eyes, brows and mouth painted out. This becomes the rig base plate.
- `assets/gen/bg_office.png` → `assets/stage/office.png`: shared stage

Prompts (abridged; full text in the commit history of this file's first version):
- Sheet: "Tall vertical 9:16 portrait. EXACTLY the same art style as the first reference image (clean dark-brown outlines, flat cel shading, one soft shadow tone, premium adult TV-animation shapes), same framing (head to belt, head in upper third, facing camera, arms relaxed, hands just out of frame). <subject description>. Plain solid flat magenta #FF00FF background. No text."
- Blank face: "Edit the reference image. Keep EVERYTHING identical … only remove both eyes, both eyebrows and the mouth, filling with matching skin, so the nose, ears and wrinkles remain."
