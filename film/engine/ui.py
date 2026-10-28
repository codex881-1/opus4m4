"""Shared on-screen graphic language: captions, labels, leaders, source tags, counters."""
import math
import skia
from . import sk
from .core import PAPER, AMBER, RED, INK, clamp, ease, E_OUT, E_SOFT, remap

MONO = "mono"


def fade_in_out(t, t0, t1, fi=0.35, fo=0.35):
    return clamp(remap(t, t0, t0 + fi)) * (1 - clamp(remap(t, t1 - fo, t1)))


def label(c, text, x, y, size=19, rgb=PAPER, a=1.0, align="left", font_name="mono", tracking=0.14,
          reveal=1.0):
    f = sk.font(font_name, size)
    return sk.draw_text(c, text, x, y, f, sk.paint(rgb, a), tracking=tracking, align=align, reveal=reveal)


def cjk(c, text, x, y, size=26, rgb=PAPER, a=1.0, align="left", tracking=0.08, weight="cjk"):
    f = sk.font(weight, size)
    return sk.draw_text(c, text, x, y, f, sk.paint(rgb, a), tracking=tracking, align=align)


def bilingual(c, en, zh, x, y, size=19, rgb=PAPER, a=1.0, align="left", zh_rgb=None):
    """English mono caps label with the Chinese name set beneath in serif CJK."""
    label(c, en, x, y, size, rgb, a, align)
    if zh:
        cjk(c, zh, x, y + size * 1.45, size * 1.05, zh_rgb or rgb, a * 0.8, align)


def caption(c, t, text, x, y, t0, t1, size=21, rgb=PAPER, align="left", font_name="mono", tracking=0.16):
    """Typed-on caption with a short underline tick; fades out at t1."""
    a = fade_in_out(t, t0, t1, 0.2, 0.4)
    if a <= 0:
        return
    rv = clamp(remap(t, t0, t0 + 0.022 * len(text) + 0.15))
    label(c, text, x, y, size, rgb, a, align, font_name, tracking, reveal=rv)


def leader(c, p0, p1, t, t0, dur=0.5, rgb=PAPER, a=0.8, width=1.3, dot=3.5):
    """Animated leader line from anchor p0 (with dot) to label point p1."""
    k = ease(remap(t, t0, t0 + dur), E_OUT)
    if k <= 0:
        return
    c.drawCircle(p0[0], p0[1], dot * min(1, k * 3), sk.paint(rgb, a))
    x = p0[0] + (p1[0] - p0[0]) * k
    y = p0[1] + (p1[1] - p0[1]) * k
    c.drawLine(p0[0], p0[1], x, y, sk.paint(rgb, a, stroke=width))


def source_tag(c, t, text, t0, t1, w=1920, h=1080, a=0.55):
    al = fade_in_out(t, t0, t1, 0.6, 0.6) * a
    if al <= 0:
        return
    label(c, "SOURCE", 64, h - 58, 13, PAPER, al * 0.7, tracking=0.3)
    label(c, text, 64 + 82, h - 58, 13, PAPER, al, tracking=0.12)


def corner_marks(c, x, y, w, h, L=18, rgb=PAPER, a=0.6, width=1.2):
    p = sk.paint(rgb, a, stroke=width, cap="butt")
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        c.drawLine(cx, cy, cx + L * dx, cy, p)
        c.drawLine(cx, cy, cx, cy + L * dy, p)


def dim_line(c, p0, p1, text, t, t0, rgb=PAPER, a=0.85, off=14, size=15):
    """Engineering dimension line with end ticks and centred value."""
    k = ease(remap(t, t0, t0 + 0.6), E_OUT)
    if k <= 0:
        return
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy / L * off, dx / L * off
    a0 = (x0 + nx, y0 + ny)
    a1 = (x0 + nx + dx * k, y0 + ny + dy * k)
    p = sk.paint(rgb, a * k, stroke=1.1)
    c.drawLine(*a0, *a1, p)
    for (px, py) in (a0, (x1 + nx, y1 + ny)):
        c.drawLine(px - nx * 0.5, py - ny * 0.5, px + nx * 0.5, py + ny * 0.5, p)
    if k > 0.6:
        mx, my = (x0 + x1) / 2 + nx * 2.1, (y0 + y1) / 2 + ny * 2.1
        c.save()
        ang = math.degrees(math.atan2(dy, dx))
        if ang > 90 or ang < -90:
            ang += 180
        c.translate(mx, my)
        c.rotate(ang)
        label(c, text, 0, size * 0.35, size, rgb, a * clamp((k - 0.6) / 0.4), align="center", tracking=0.1)
        c.restore()


def number_roll(value, k):
    """Counter that decelerates into its final value."""
    return int(round(value * ease(k, E_SOFT)))
