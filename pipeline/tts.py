"""TTS providers. Each returns (wav_path, words[{w,t0,t1}], provider_label).

edge-tts: Microsoft neural voices, with word-boundary timings.
voxcpm:   VoxCPM2 voice design (prompted voice description), run in .venv-vox; no word timings,
          so words are aligned later from Rhubarb/energy (see align_words_by_energy).
"""
import asyncio, json, os, subprocess, wave, shutil
import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SR = 24000

def _to_wav(src, dst, sr=SR):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", dst], check=True)

def edge(text, voice, out_wav, rate="+0%", pitch="+0Hz"):
    import edge_tts
    words = []
    mp3 = out_wav[:-4] + ".mp3"
    async def go():
        c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
        with open(mp3, "wb") as f:
            async for ch in c.stream():
                if ch["type"] == "audio":
                    f.write(ch["data"])
                elif ch["type"] == "WordBoundary":
                    t0 = ch["offset"] / 1e7; d = ch["duration"] / 1e7
                    words.append({"w": ch["text"], "t0": round(t0, 3), "t1": round(t0 + d, 3)})
    for attempt in range(3):
        try:
            asyncio.run(go()); break
        except Exception as e:  # network hiccup
            if attempt == 2: raise
    _to_wav(mp3, out_wav); os.remove(mp3)
    return out_wav, words, "edge-tts:" + voice

def voxcpm(text, control, out_wav, seed=7, cfg=2.0, steps=10):
    py = os.path.join(ROOT, ".venv-vox", "bin", "python")
    script = os.path.join(ROOT, "pipeline", "vox_worker.py")
    raw = out_wav[:-4] + ".raw.wav"
    r = subprocess.run([py, script, "--text", text, "--control", control, "--seed", str(seed),
                        "--cfg", str(cfg), "--steps", str(steps), "--out", raw],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("voxcpm failed: " + (r.stderr or r.stdout)[-1500:])
    _to_wav(raw, out_wav); os.remove(raw)
    return out_wav, [], "voxcpm2-design:" + control[:60]

def trim_silence(wav, thresh_db=-42, pad=0.05):
    a, sr = sf.read(wav)
    env = np.abs(a)
    win = int(sr * 0.01)
    e = np.convolve(env, np.ones(win) / win, mode="same")
    thr = 10 ** (thresh_db / 20)
    idx = np.where(e > thr)[0]
    if len(idx) == 0: return 0.0
    s = max(0, idx[0] - int(pad * sr)); t = min(len(a), idx[-1] + int(pad * sr))
    sf.write(wav, a[s:t], sr)
    return s / sr

def align_words_by_energy(wav, text):
    """Fallback word timing when the TTS gives none: distribute words over voiced spans,
    weighting by character length. Good enough for captions and emphasis nods."""
    a, sr = sf.read(wav)
    dur = len(a) / sr
    ws = text.split()
    total = sum(len(w) + 2 for w in ws)
    t = 0.0; out = []
    for w in ws:
        d = dur * (len(w) + 2) / total
        out.append({"w": w, "t0": round(t, 3), "t1": round(t + d * 0.9, 3)}); t += d
    return out

def duration(wav):
    with wave.open(wav) as w:
        return w.getnframes() / w.getframerate()

# ---------------- Kokoro-82M (local, CPU, Apache-2.0) ----------------
_KOK = None
def _kok():
    global _KOK
    if _KOK is None:
        from kokoro_onnx import Kokoro
        d = os.path.join(ROOT, ".cache", "kokoro")
        _KOK = Kokoro(os.path.join(d, "kokoro-v1.0.onnx"), os.path.join(d, "voices-v1.0.bin"))
    return _KOK

def _syll(w):
    import re
    w = re.sub(r"[^a-z]", "", w.lower()) or "a"
    n = len(re.findall(r"[aeiouy]+", w)); n -= 1 if w.endswith("e") and n > 1 else 0
    return max(1, n)

def kokoro(text, voice, out_wav, speed=1.0, lang="en-us"):
    """voice: 'am_onyx' or blend 'am_onyx:0.6+am_michael:0.4'. Synth per sentence so we know
    sentence timing; words are spread inside a sentence by syllable weight."""
    import re
    k = _kok()
    if "+" in voice or ":" in voice:
        st = None
        for part in voice.split("+"):
            n, _, wgt = part.partition(":"); s = k.get_voice_style(n) * float(wgt or 1)
            st = s if st is None else st + s
        v = st
    else:
        v = voice
    sents = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]
    pieces, words, t = [], [], 0.0
    sr_out = 24000
    for si, s in enumerate(sents):
        a, sr = k.create(s, voice=v, speed=speed, lang=lang)
        sr_out = sr
        e = np.abs(a); thr = 0.01
        idx = np.where(np.convolve(e, np.ones(240) / 240, mode="same") > thr)[0]
        if len(idx): a = a[max(0, idx[0] - 240): idx[-1] + 480]
        d = len(a) / sr
        ws = s.split(); wt = [_syll(w) + 0.6 for w in ws]; tot = sum(wt); tt = t + 0.03
        span = d - 0.08
        for w, x in zip(ws, wt):
            dd = span * x / tot; words.append({"w": w, "t0": round(tt, 3), "t1": round(tt + dd * 0.92, 3)}); tt += dd
        pieces.append(a); t += d
        if si < len(sents) - 1:
            gap = 0.22 if s.endswith((".", "!", "?")) else 0.1
            pieces.append(np.zeros(int(gap * sr))); t += gap
    a = np.concatenate(pieces).astype(np.float32)
    tmp = out_wav[:-4] + ".k.wav"; sf.write(tmp, a, sr_out); _to_wav(tmp, out_wav); os.remove(tmp)
    return out_wav, words, "kokoro-82m:" + voice
