#!/usr/bin/env bash
# Re-provision the sandbox (system pip/apt packages vanish between sessions; /workspace persists).
set -u
cd "$(dirname "$0")"
export TMPDIR=$PWD/.cache/tmp PIP_CACHE_DIR=$PWD/.cache/pip; mkdir -p "$TMPDIR"
python3 -c "import playwright, kokoro_onnx, scipy, soundfile, edge_tts, numpy" 2>/dev/null || \
  pip install -q playwright kokoro-onnx scipy soundfile edge-tts numpy
command -v chromium >/dev/null || { sudo apt-get update -q >/dev/null; sudo apt-get install -y -q chromium >/dev/null; }  # playwright's own arm64 chromium download is 403
command -v ffmpeg >/dev/null || sudo apt-get install -y -q ffmpeg >/dev/null
# rhubarb binary is built into vendors/ (persists); rebuild needs: cmake build-essential libboost-all-dev
[ -x vendors/rhubarb-lip-sync/build/rhubarb/rhubarb ] || echo "rhubarb missing: see experiments/MANIFEST.md"
# fonts + kokoro model live in assets/fonts and .cache/kokoro (persist)
echo "setup ok"
