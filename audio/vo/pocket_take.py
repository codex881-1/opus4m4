"""Render narration lines with Kyutai Pocket TTS. Args: lines.json outdir voices [temp] [seeds]"""
import sys, json, os, time, torch, numpy as np, soundfile as sf
from pocket_tts import TTSModel
lines = json.load(open(sys.argv[1])); outdir = sys.argv[2]; voices = sys.argv[3].split(',')
temp = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
seeds = [int(s) for s in sys.argv[5].split(',')] if len(sys.argv) > 5 else [0]
only = [int(i) for i in sys.argv[6].split(',')] if len(sys.argv) > 6 else None
os.makedirs(outdir, exist_ok=True)
model = TTSModel.load_model()
for v in voices:
    st = model.get_state_for_audio_prompt(v)
    for i, text in enumerate(lines):
        if only and i not in only: continue
        for sd in seeds:
            torch.manual_seed(sd)
            t = time.time()
            kw = {"temperature": temp} if temp is not None else {}
            try:
                audio = model.generate_audio(st, text, **kw)
            except TypeError:
                audio = model.generate_audio(st, text)
            a = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            sf.write(f"{outdir}/{v}_{i:02d}_s{sd}.wav", a.squeeze(), model.sample_rate)
            print(v, i, sd, round(a.shape[-1]/model.sample_rate, 2), 'in', round(time.time()-t, 1), flush=True)
