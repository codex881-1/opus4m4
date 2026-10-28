"""Re-download the Freesound recordings used in the mix (all CC0) and convert to 48 kHz stereo WAV.

Metadata (author, licence, source URL) for each file lives next to it as rec/<id>.json.
Downloads are sequential with a pause, using the public HQ preview (OGG, falling back to MP3).
"""
import glob
import json
import os
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "rec")

for meta_path in sorted(glob.glob(os.path.join(REC, "*.json"))):
    m = json.load(open(meta_path))
    sid = str(m["id"])
    wav = os.path.join(REC, sid + ".wav")
    if os.path.exists(wav):
        continue
    mp3 = m["hq_mp3"]
    for url in (mp3.replace("-hq.mp3", "-hq.ogg"), mp3):
        tmp = os.path.join(REC, sid + os.path.splitext(url)[1])
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as e:  # try the next format
            print(sid, "failed", url, e)
            continue
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ar", "48000", "-ac", "2", wav], check=True)
        os.remove(tmp)
        print(sid, m["license"], m["title"])
        break
    time.sleep(1.5)
