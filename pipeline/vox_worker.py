"""Runs inside .venv-vox. VoxCPM2 voice design: text + natural-language voice description -> wav."""
import argparse, time, sys, os
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "hf"))
p = argparse.ArgumentParser()
p.add_argument("--text"); p.add_argument("--control"); p.add_argument("--seed", type=int, default=7)
p.add_argument("--cfg", type=float, default=2.0); p.add_argument("--steps", type=int, default=10)
p.add_argument("--out")
a = p.parse_args()
import torch, numpy as np, soundfile as sf
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(2)
from voxcpm import VoxCPM
t = time.time()
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
t_load = time.time() - t
t = time.time()
# VoxCPM2 voice design: description in parentheses before the text
wav = model.generate(text="(%s)%s" % (a.control, a.text), cfg_value=a.cfg, inference_timesteps=a.steps)
sr = getattr(getattr(model, "tts_model", model), "sample_rate", 48000)
sf.write(a.out, wav, sr)
print("load_s=%.1f gen_s=%.1f audio_s=%.2f sr=%d" % (t_load, time.time() - t, len(wav) / sr, sr), file=sys.stderr)
