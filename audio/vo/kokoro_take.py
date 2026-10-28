"""Render narration lines with Kokoro; one wav per line per voice."""
import sys, json, os, numpy as np, soundfile as sf
from kokoro import KPipeline
lines = json.load(open(sys.argv[1]))
outdir = sys.argv[2]; voices = sys.argv[3].split(','); speed = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
os.makedirs(outdir, exist_ok=True)
pipe = KPipeline(lang_code='a')
for v in voices:
    for i, text in enumerate(lines):
        chunks = [a for _, _, a in pipe(text, voice=v, speed=speed)]
        audio = np.concatenate([c.numpy() if hasattr(c, 'numpy') else c for c in chunks])
        sf.write(f"{outdir}/{v}_{i:02d}.wav", audio, 24000)
        print(v, i, round(len(audio)/24000, 2), flush=True)
