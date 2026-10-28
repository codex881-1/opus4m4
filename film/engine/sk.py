"""Skia vector + typography helpers. All canvases are premultiplied RGBA8, top-left origin."""
import functools
import math
import os
import numpy as np
import skia

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")

FONTS = {
    "mono": "IBMPlexMono-Regular.ttf",
    "mono_l": "IBMPlexMono-Light.ttf",
    "mono_m": "IBMPlexMono-Medium.ttf",
    "mono_sb": "IBMPlexMono-SemiBold.ttf",
    "cond": "IBMPlexSansCondensed-Regular.ttf",
    "cond_l": "IBMPlexSansCondensed-Light.ttf",
    "cond_m": "IBMPlexSansCondensed-Medium.ttf",
    "cond_sb": "IBMPlexSansCondensed-SemiBold.ttf",
    "serif": "IBMPlexSerif-Regular.ttf",
    "serif_l": "IBMPlexSerif-Light.ttf",
    "serif_i": "IBMPlexSerif-Italic.ttf",
    "display": "InstrumentSerif-Regular.ttf",
    "display_i": "InstrumentSerif-Italic.ttf",
    "type": "CourierPrime-Regular.ttf",
    "type_b": "CourierPrime-Bold.ttf",
}
CJK_SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
CJK_SERIF_R = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"


@functools.lru_cache(None)
def typeface(name):
    if name == "cjk":
        return skia.Typeface.MakeFromFile(CJK_SERIF, 0)
    if name == "cjk_r":
        return skia.Typeface.MakeFromFile(CJK_SERIF_R if os.path.exists(CJK_SERIF_R) else CJK_SERIF, 0)
    return skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, FONTS[name]))


def font(name, size):
    f = skia.Font(typeface(name), size)
    f.setSubpixel(True)
    f.setLinearMetrics(True)
    f.setHinting(skia.FontHinting.kNone)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    return f


def color(rgb, a=1.0):
    r, g, b = [max(0, min(255, int(round(c * 255)))) for c in rgb]
    return skia.Color(r, g, b, max(0, min(255, int(round(a * 255)))))


def paint(rgb=(1, 1, 1), a=1.0, stroke=None, cap="round", join="round", blur=0.0, aa=True):
    p = skia.Paint(AntiAlias=aa, Color=color(rgb, a))
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap({"round": skia.Paint.kRound_Cap, "butt": skia.Paint.kButt_Cap,
                        "square": skia.Paint.kSquare_Cap}[cap])
        p.setStrokeJoin({"round": skia.Paint.kRound_Join, "miter": skia.Paint.kMiter_Join,
                         "bevel": skia.Paint.kBevel_Join}[join])
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


class Canvas:
    """A numpy-backed skia surface."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.arr = np.zeros((h, w, 4), np.uint8)
        self.surface = skia.Surface(self.arr, colorType=skia.kRGBA_8888_ColorType,
                                    alphaType=skia.kPremul_AlphaType)
        self.c = self.surface.getCanvas()

    def clear(self, rgb=None, a=0.0):
        self.c.clear(color(rgb or (0, 0, 0), a))
        return self

    def flush(self):
        self.surface.flushAndSubmit()
        return self.arr


# ------------------------------------------------------------------ text
def text_width(s, f, tracking=0.0):
    if not s:
        return 0.0
    glyphs = f.textToGlyphs(s)
    ws = f.getWidths(glyphs)
    return float(sum(ws) + tracking * f.getSize() * (len(ws) - 1))


def draw_text(c, s, x, y, f, p, tracking=0.0, align="left", reveal=1.0, per_char=None):
    """Draw text with tracking (em fraction). reveal<1 types characters on.
    per_char(i, n) -> (dx, dy, alpha) optional jitter callback."""
    if not s:
        return 0.0
    glyphs = f.textToGlyphs(s)
    ws = f.getWidths(glyphs)
    tr = tracking * f.getSize()
    total = sum(ws) + tr * (len(ws) - 1)
    if align == "center":
        x -= total / 2
    elif align == "right":
        x -= total
    n = len(glyphs)
    shown = n if reveal >= 1 else int(math.floor(reveal * n + 1e-6))
    xs = []
    cx = x
    for w in ws:
        xs.append(cx)
        cx += w + tr
    if per_char is None:
        if shown <= 0:
            return total
        blob = skia.TextBlob.MakeFromPosTextH(s[:shown], xs[:shown], y, f) if shown == len(s) else None
        if blob is None:
            builder = skia.TextBlobBuilder()
            builder.allocRunPosH(f, list(glyphs[:shown]), xs[:shown], y)
            blob = builder.make()
        c.drawTextBlob(blob, 0, 0, p)
        return total
    base_a = p.getAlphaf()
    for i in range(shown):
        dx, dy, a = per_char(i, n)
        q = skia.Paint(p)
        q.setAlphaf(base_a * a)
        builder = skia.TextBlobBuilder()
        builder.allocRunPosH(f, [glyphs[i]], [xs[i] + dx], y + dy)
        c.drawTextBlob(builder.make(), 0, 0, q)
    return total


# ------------------------------------------------------------------ paths
def polyline(points, closed=False):
    path = skia.Path()
    if len(points) == 0:
        return path
    path.moveTo(*map(float, points[0]))
    for p in points[1:]:
        path.lineTo(*map(float, p))
    if closed:
        path.close()
    return path


def smooth_path(points, tension=0.5, closed=False):
    """Catmull-Rom through points -> cubic path."""
    pts = [tuple(map(float, p)) for p in points]
    path = skia.Path()
    if len(pts) < 2:
        return path
    path.moveTo(*pts[0])
    n = len(pts)
    for i in range(n - 1 if not closed else n):
        p0 = pts[i - 1] if i > 0 or closed else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (i + 2 < n or closed) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) * tension / 3, p1[1] + (p2[1] - p0[1]) * tension / 3)
        c2 = (p2[0] - (p3[0] - p1[0]) * tension / 3, p2[1] - (p3[1] - p1[1]) * tension / 3)
        path.cubicTo(*c1, *c2, *p2)
    if closed:
        path.close()
    return path


def _contours(path):
    pm = skia.PathMeasure(path, False)
    out = []
    while True:
        L = pm.getLength()
        if L > 0:
            out.append((pm, L))
            # PathMeasure is stateful; capture segments eagerly via a fresh copy per contour
        if not pm.nextContour():
            break
    return out


def path_length(path):
    pm = skia.PathMeasure(path, False)
    total = 0.0
    while True:
        total += pm.getLength()
        if not pm.nextContour():
            break
    return total


def sub_path(path, t0, t1):
    """Portion of path between fractions t0..t1 of its total length (all contours)."""
    t0, t1 = max(0.0, t0), min(1.0, t1)
    out = skia.Path()
    if t1 <= t0:
        return out
    total = path_length(path)
    a, b = t0 * total, t1 * total
    acc = 0.0
    pm = skia.PathMeasure(path, False)
    while True:
        L = pm.getLength()
        s, e = max(a - acc, 0.0), min(b - acc, L)
        if e > s:
            seg = skia.Path()
            pm.getSegment(s, e, seg, True)
            out.addPath(seg)
        acc += L
        if acc >= b or not pm.nextContour():
            break
    return out


def point_at(path, t):
    pm = skia.PathMeasure(path, False)
    L = pm.getLength()
    if L <= 0:
        return (0.0, 0.0), (1.0, 0.0)
    pos, tan = pm.getPosTan(max(0.0, min(L, t * L)))
    return (pos.x(), pos.y()), (tan.x(), tan.y())


def glow_stroke(c, path, rgb, width, a=1.0, glow=10.0, glow_a=0.55):
    """Crisp stroke over a soft halo: reads like phosphor/ink light, not a drop shadow."""
    if glow > 0 and glow_a > 0:
        c.drawPath(path, paint(rgb, a * glow_a, stroke=width * 2.2, blur=glow))
    c.drawPath(path, paint(rgb, a, stroke=width))


def dashed(p, on, off, phase=0.0):
    p.setPathEffect(skia.DashPathEffect.Make([on, off], phase))
    return p


def to_image(arr):
    return skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType)
