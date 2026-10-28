"""Contact sheet: python tools/sheet.py DIR out.jpg f1 f2 f3 f4 (frame numbers)."""
import sys
from PIL import Image

d, out = sys.argv[1], sys.argv[2]
frames = [int(x) for x in sys.argv[3:]]
cols = 2
w, h = 960, 540
rows = (len(frames) + 1) // 2
g = Image.new("RGB", (w * cols, h * rows))
for i, n in enumerate(frames):
    im = Image.open(f"{d}/f{n:05d}.png").resize((w, h), Image.LANCZOS)
    g.paste(im, ((i % cols) * w, (i // cols) * h))
g.save(out, quality=90)
