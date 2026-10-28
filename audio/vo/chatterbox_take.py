"""Render narration lines with Chatterbox (CPU). Args: lines.json outdir [exaggeration cfg seeds audio_prompt]"""
import sys, json, os, time, torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
lines = json.load(open(sys.argv[1])); outdir = sys.argv[2]
ex = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
cfg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
seeds = [int(s) for s in sys.argv[5].split(',')] if len(sys.argv) > 5 else [1]
prompt = sys.argv[6] if len(sys.argv) > 6 else None
only = [int(i) for i in sys.argv[7].split(',')] if len(sys.argv) > 7 else None
os.makedirs(outdir, exist_ok=True)
torch.set_num_threads(2)
model = ChatterboxTTS.from_pretrained(device="cpu")
for i, text in enumerate(lines):
    if only and i not in only: continue
    for sd in seeds:
        torch.manual_seed(sd)
        t = time.time()
        wav = model.generate(text, audio_prompt_path=prompt, exaggeration=ex, cfg_weight=cfg)
        ta.save(f"{outdir}/cb_{i:02d}_s{sd}.wav", wav, model.sr)
        print(i, sd, round(wav.shape[-1]/model.sr, 2), 'sec audio in', round(time.time()-t, 1), 's', flush=True)
