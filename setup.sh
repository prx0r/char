#!/usr/bin/env bash
# Re-provision the sandbox (system pip/apt packages vanish between sessions; /workspace persists).
set -u
cd "$(dirname "$0")"
export TMPDIR=$PWD/.cache/tmp PIP_CACHE_DIR=$PWD/.cache/pip; mkdir -p "$TMPDIR"
python3 -c "import playwright, kokoro_onnx, scipy, soundfile, edge_tts, numpy" 2>/dev/null || \
  pip install -q playwright kokoro-onnx scipy soundfile edge-tts numpy
command -v chromium >/dev/null || { sudo apt-get update -q >/dev/null; sudo apt-get install -y -q chromium >/dev/null; }  # playwright's own arm64 chromium download is 403
command -v ffmpeg >/dev/null || sudo apt-get install -y -q ffmpeg >/dev/null
# rigkit: MediaPipe (pose/face landmarks for onboarding; its C lib needs libEGL) + opencv comes with it
python3 -c "import mediapipe" 2>/dev/null || pip install -q mediapipe
ldconfig -p | grep -q libEGL.so.1 || sudo apt-get install -y -q libegl1 libgles2 >/dev/null
mkdir -p .cache/mediapipe
[ -f .cache/mediapipe/face_landmarker.task ] || curl -sSLo .cache/mediapipe/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
[ -f .cache/mediapipe/pose_landmarker_full.task ] || curl -sSLo .cache/mediapipe/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
# rhubarb binary is built into vendors/ (persists); rebuild needs: cmake build-essential libboost-all-dev
[ -x vendors/rhubarb-lip-sync/build/rhubarb/rhubarb ] || echo "rhubarb missing: see experiments/MANIFEST.md"
# fonts + kokoro model live in assets/fonts and .cache/kokoro (persist)
echo "setup ok"
