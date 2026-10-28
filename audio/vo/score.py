"""Score narration takes: UTMOS (naturalness MOS), Whisper WER vs script, pitch variability, rate."""
import sys, json, glob, re, numpy as np, torch, librosa
from faster_whisper import WhisperModel
torch.set_num_threads(2)
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = np.zeros((len(r)+1, len(h)+1)); d[:,0] = range(len(r)+1); d[0,:] = range(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(r[i-1]!=h[j-1]))
    return d[-1,-1]/max(len(r),1)
files = sys.argv[2:]; refs = json.load(open(sys.argv[1]))
for f in files:
    idx = int(re.findall(r"_(\d\d)", f.split('/')[-1])[0])
    y, sr = librosa.load(f, sr=16000)
    mos = utmos(torch.tensor(y)[None], 16000).item()
    segs, _ = asr.transcribe(y.astype("float32"), language="en", beam_size=5)
    hyp = " ".join(s.text for s in segs)
    f0, vf, _ = librosa.pyin(y, fmin=70, fmax=400, sr=16000)
    f0v = f0[~np.isnan(f0)]
    st = 12*np.log2(f0v/np.median(f0v)) if len(f0v) else np.array([0])
    print(json.dumps({"file": f.split('/')[-1], "mos": round(mos,3), "wer": round(wer(refs[idx], hyp),3), "f0_med": round(float(np.median(f0v)),1) if len(f0v) else 0,
        "f0_sd_semitones": round(float(st.std()),2), "dur": round(len(y)/16000,2), "wps": round(len(refs[idx].split())/(len(y)/16000),2), "hyp": hyp.strip()}), flush=True)
