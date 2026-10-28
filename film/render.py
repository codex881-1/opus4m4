"""Master renderer: composites the scene timeline, applies transitions + film post, pipes to ffmpeg.

Usage: python render.py --start F0 --end F1 --out seg.mkv [--scale 0.5] [--png DIR --every N]
"""
import argparse
import os
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import FPS, W, H, lerp  # noqa: E402
from engine.gl import GL, POST_DEFAULT  # noqa: E402


def blend_post(a, b, k):
    out = {}
    for key in set(a) | set(b):
        va, vb = a.get(key, POST_DEFAULT.get(key)), b.get(key, POST_DEFAULT.get(key))
        if isinstance(va, (tuple, list)):
            out[key] = tuple(lerp(x, y, k) for x, y in zip(va, vb))
        else:
            out[key] = lerp(va, vb, k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--frames", default=None, help="comma list of frame numbers (png mode)")
    ap.add_argument("--only", default=None, help="comma list of scene names to build")
    args = ap.parse_args()

    from scenes.timeline import build_timeline
    gl = GL(W, H)
    tl = build_timeline(gl, only=args.only.split(",") if args.only else None)
    total = tl.total_frames()
    end = total if args.end is None else min(args.end, total)
    frames = [int(x) for x in args.frames.split(",")] if args.frames else list(range(args.start, end, args.every))

    proc = None
    if args.out:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p", args.out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if args.png:
        os.makedirs(args.png, exist_ok=True)
        from PIL import Image

    t0 = time.time()
    for n, fi in enumerate(frames):
        T = fi / FPS
        hdr, post = tl.render(T)
        out = gl.post(hdr, T, **post)
        data = gl.read_rgb(out)
        if proc:
            proc.stdin.write(data)
        if args.png:
            img = np.frombuffer(data, np.uint8).reshape(H, W, 3)
            Image.fromarray(img).save(os.path.join(args.png, f"f{fi:05d}.png"))
        if n % 24 == 0:
            el = time.time() - t0
            print(f"frame {fi} ({n + 1}/{len(frames)}) {el / (n + 1):.2f}s/f", flush=True)
    if proc:
        proc.stdin.close()
        proc.wait()
    print("done", len(frames), "frames in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
