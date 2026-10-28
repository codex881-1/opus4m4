"""Assemble the narration: best take per line -> placed on the edit grid -> word timings for picture sync.

Writes audio/vo/narration.wav (48 kHz mono) and audio/vo/timing.json:
  {"lines": [{"i", "start", "end", "file", "text"}], "words": [{"w", "t0", "t1", "line"}]}
"""
import json
import os
import re
import sys
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from faster_whisper import WhisperModel

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 48000
# planned line starts (s) on the 69.9 s edit grid
STARTS = [1.0, 7.5, 9.7, 15.2, 20.3, 26.7, 31.8, 39.8, 46.7, 51.5, 54.9, 60.8]
TOTAL = 69.9


def best_takes():
    rows = []
    for f in (os.path.join(ROOT, "scores", "jane_seeds.jsonl"), os.path.join(ROOT, "scores", "all_voices.jsonl")):
        for l in open(f):
            if l.startswith('{"file": "jane'):
                rows.append(json.loads(l))
    best = {}
    for r in rows:
        i = int(r["file"].split("_")[1])
        if i not in best or r["mos"] > best[i]["mos"]:
            best[i] = r
    return best


def trim(y, sr, top_db=38, pad=0.03):
    idx = librosa.effects.split(y, top_db=top_db)
    a, b = idx[0][0], idx[-1][1]
    a = max(0, a - int(pad * sr))
    b = min(len(y), b + int(pad * 1.5 * sr))
    return y[a:b]


def voice_chain(y):
    """Gentle broadcast chain: HPF 80 Hz, de-box dip at 300 Hz, presence lift ~3.2 kHz, soft air shelf."""
    sos = signal.butter(2, 80, "hp", fs=SR, output="sos")
    y = signal.sosfilt(sos, y)

    def peak(y, f0, gain_db, q):
        A = 10 ** (gain_db / 40)
        w = 2 * np.pi * f0 / SR
        al = np.sin(w) / (2 * q)
        b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
        a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
        return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], y)
    y = peak(y, 300, -1.5, 1.0)
    y = peak(y, 3200, 1.8, 0.9)
    y = peak(y, 9000, 1.2, 0.7)
    return y


def main():
    lines = json.load(open(os.path.join(ROOT, "script.json")))
    best = best_takes()
    asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    out = np.zeros(int(TOTAL * SR) + SR)
    meta = {"lines": [], "words": []}
    prev_end = 0.0
    for i, text in enumerate(lines):
        f = os.path.join(ROOT, "takes", "jane" if "_s0" not in best[i]["file"] else "pocket", best[i]["file"])
        if not os.path.exists(f):
            f = os.path.join(ROOT, "takes", "pocket", best[i]["file"])
        y, sr = librosa.load(f, sr=SR)
        y = trim(y, SR)
        y = voice_chain(y)
        y /= np.max(np.abs(y)) + 1e-9
        fade = int(0.012 * SR)
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
        st = STARTS[i]
        if st < prev_end + 0.15:
            print(f"WARNING line {i} overlaps previous (start {st}, prev end {prev_end:.2f})")
        a = int(st * SR)
        out[a:a + len(y)] += y * 0.5
        end = st + len(y) / SR
        prev_end = end
        y16 = librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=16000)
        segs, _ = asr.transcribe(y16, language="en", word_timestamps=True, beam_size=5)
        for s in segs:
            for w in s.words:
                meta["words"].append({"w": re.sub(r"[^\w']", "", w.word.lower()), "t0": round(st + w.start, 3),
                                      "t1": round(st + w.end, 3), "line": i})
        meta["lines"].append({"i": i, "start": st, "end": round(end, 3), "file": os.path.basename(f), "text": text,
                              "mos": best[i]["mos"]})
        print(f"L{i:02d} {st:6.2f}-{end:6.2f}  {best[i]['file']}  mos={best[i]['mos']}")
    out = out[: int(TOTAL * SR)]
    sf.write(os.path.join(ROOT, "narration.wav"), out.astype(np.float32), SR, subtype="FLOAT")
    json.dump(meta, open(os.path.join(ROOT, "timing.json"), "w"), indent=1)
    print("total", TOTAL, "last line ends", prev_end)


if __name__ == "__main__":
    main()
