"""Declassified-document system: procedural paper, typewriter ink, stamps, redactions, raking lamp light."""
import math
import os
import numpy as np
import moderngl
import skia
from PIL import Image
from scipy import ndimage
from . import sk
from .core import hash1, clamp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def _fbm(h, w, seed, octaves=6, base=4):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = base * 2 ** o, int(base * 2 ** o * w / h) + 1
        g = r.standard_normal((gh + 1, gw + 1)).astype(np.float32)
        z = ndimage.zoom(g, (h / gh, w / gw), order=3)[:h, :w]
        out += z * amp
        tot += amp
        amp *= 0.55
    return out / tot


def make_paper(w=2048, h=2650, seed=7, tone=(0.905, 0.872, 0.795), key="paper"):
    """Returns (albedo uint8 HxWx3, height float32 HxW). Cached on disk."""
    fa = os.path.join(CACHE, f"{key}2_{w}x{h}_{seed}_a.png")
    fh = os.path.join(CACHE, f"{key}2_{w}x{h}_{seed}_h.npy")
    if os.path.exists(fa) and os.path.exists(fh):
        return np.asarray(Image.open(fa)), np.load(fh)
    r = np.random.default_rng(seed)
    mott = _fbm(h, w, seed, octaves=5, base=3)
    tooth = ndimage.gaussian_filter(r.standard_normal((h, w)).astype(np.float32), 0.8)
    # fibres: thousands of short faint curved strokes
    cv = sk.Canvas(w, h).clear((0, 0, 0), 0)
    for i in range(9000):
        x, y = r.uniform(0, w), r.uniform(0, h)
        ang = r.uniform(0, math.pi)
        L = r.uniform(6, 38)
        bend = r.uniform(-0.4, 0.4)
        pts = []
        for k in range(6):
            t = k / 5
            a = ang + bend * t
            pts.append((x + math.cos(a) * L * t, y + math.sin(a) * L * t))
        p = sk.paint((1, 1, 1), r.uniform(0.05, 0.22), stroke=r.uniform(0.5, 1.3))
        cv.c.drawPath(sk.smooth_path(pts), p)
    fib = cv.flush()[..., 3].astype(np.float32) / 255.0
    # age: edges slightly darker/warmer, faint fold creases (letter fold in thirds)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum.reduce([xx, w - xx, yy, h - yy]) / (0.08 * w)
    edge = np.clip(edge, 0, 1)
    crease = np.zeros((h, w), np.float32)
    for fy in (h / 3, 2 * h / 3):
        d = np.abs(yy - fy - (xx - w / 2) * 0.004)
        crease += np.exp(-(d / 9.0) ** 2) * 0.6 - np.exp(-((d - 14) / 16.0) ** 2) * 0.2
    height = mott * 0.5 + tooth * 0.05 + fib * 0.22 - crease * 0.35
    lum = 1.0 + mott * 0.045 + tooth * 0.018 - fib * 0.03 - (1 - edge) * 0.07 - np.clip(crease, 0, None) * 0.015
    warm = (1 - edge) * 0.05
    alb = np.stack([tone[0] * lum, tone[1] * lum * (1 - warm * 0.4), tone[2] * lum * (1 - warm)], -1)
    alb = np.clip(alb * 255, 0, 255).astype(np.uint8)
    Image.fromarray(alb).save(fa)
    np.save(fh, height.astype(np.float32))
    return alb, height.astype(np.float32)


VS_DOC = """
#version 330
in vec3 in_pos; in vec2 in_uv;
uniform mat4 u_mvp; uniform mat4 u_model;
out vec2 v_uv; out vec3 v_w;
void main(){ v_uv = in_uv; v_w = (u_model * vec4(in_pos, 1.0)).xyz; gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""

FS_DOC = """
#version 330
in vec2 v_uv; in vec3 v_w; out vec4 o;
uniform sampler2D u_albedo; uniform sampler2D u_height; uniform sampler2D u_ink; uniform sampler2D u_red;
uniform vec3 u_lamp; uniform vec3 u_lamp_col; uniform vec3 u_amb; uniform float u_opacity;
uniform vec2 u_texel; uniform float u_bump; uniform float u_time; uniform float u_ink_gain;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float vnoise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }
void main(){
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    float hc = texture(u_height, uv).r;
    float hx = texture(u_height, uv + vec2(u_texel.x, 0)).r - texture(u_height, uv - vec2(u_texel.x, 0)).r;
    float hy = texture(u_height, uv + vec2(0, u_texel.y)).r - texture(u_height, uv - vec2(0, u_texel.y)).r;
    vec3 n = normalize(vec3(-hx * u_bump, hy * u_bump, 1.0));
    vec3 alb = texture(u_albedo, uv).rgb;
    // typewriter ink: uneven ribbon density, slightly fibrous edges
    float ink = texture(u_ink, uv).a;
    float rib = 0.86 + 0.22 * vnoise(uv * vec2(900.0, 1150.0)) - 0.08 * vnoise(uv * vec2(60.0, 4.0));
    float edge = smoothstep(0.05, 0.42, ink * (0.85 + 0.3 * vnoise(uv * 2400.0)));
    float inkA = clamp(edge * rib * u_ink_gain, 0.0, 1.0);
    vec3 inkCol = vec3(0.035, 0.032, 0.045);
    alb = mix(alb, inkCol, inkA * 0.97);
    // stamp ink (red, starved / mottled)
    vec4 red = texture(u_red, uv);
    float starve = smoothstep(0.25, 0.75, vnoise(uv * 340.0) * 0.6 + vnoise(uv * 41.0) * 0.6);
    float ra = red.a * (0.45 + 0.6 * starve);
    vec3 redCol = red.a > 0.0 ? red.rgb / max(red.a, 1e-3) : vec3(0);
    alb = mix(alb, alb * redCol * 1.15, clamp(ra, 0.0, 1.0));
    // raking lamp: point light close to the page
    vec3 L = u_lamp - v_w; float d = length(L); L /= d;
    float ndl = max(dot(n, L), 0.0);
    float fall = 1.0 / (1.0 + 0.9 * d * d);
    vec3 col = alb * (u_amb + u_lamp_col * ndl * fall);
    o = vec4(col, 1.0) * u_opacity;
}
"""


class PaperGL:
    """A lit sheet of paper in 3D, with ink + stamp layers supplied as skia canvases."""

    def __init__(self, gl, w=2048, h=2650, seed=7, tone=(0.905, 0.872, 0.795)):
        self.gl = gl
        ctx = gl.ctx
        alb, hgt = make_paper(w, h, seed, tone)
        self.w, self.h = w, h
        self.alb = ctx.texture((w, h), 3, data=np.ascontiguousarray(alb).tobytes())
        self.alb.build_mipmaps()
        self.alb.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.alb.anisotropy = 8.0
        self.hgt = ctx.texture((w, h), 1, data=np.ascontiguousarray(hgt).tobytes(), dtype="f4")
        self.hgt.filter = (moderngl.LINEAR, moderngl.LINEAR)
        for tx in (self.alb, self.hgt):
            tx.repeat_x = tx.repeat_y = False
        if not hasattr(gl, "_doc_prog"):
            gl._doc_prog = ctx.program(vertex_shader=VS_DOC, fragment_shader=FS_DOC)
            gl._doc_vao = ctx.vertex_array(gl._doc_prog, [(gl.plane_vbo, "3f 2f", "in_pos", "in_uv")])
        self.prog, self.vao = gl._doc_prog, gl._doc_vao
        self.ink = Canvas2(w, h)
        self.red = Canvas2(w, h)
        self.ink_tex = None
        self.red_tex = None

    def upload(self):
        self.ink_tex = self.gl.texture_rgba8(self.ink.cv.flush(), mipmap=True, tex=self.ink_tex)
        self.red_tex = self.gl.texture_rgba8(self.red.cv.flush(), mipmap=True, tex=self.red_tex)

    def draw(self, mvp, model, lamp=(0.2, 0.5, 0.6), lamp_col=(1.6, 1.45, 1.2), amb=(0.10, 0.10, 0.12),
             opacity=1.0, bump=0.9, ink_gain=1.0):
        ctx = self.gl.ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        p = self.prog
        for i, (name, t) in enumerate([("u_albedo", self.alb), ("u_height", self.hgt), ("u_ink", self.ink_tex),
                                       ("u_red", self.red_tex)]):
            t.use(i)
            p[name].value = i
        p["u_mvp"].write(np.asarray(mvp, "f4").T.tobytes())
        p["u_model"].write(np.asarray(model, "f4").T.tobytes())
        p["u_lamp"].value = tuple(map(float, lamp))
        p["u_lamp_col"].value = tuple(map(float, lamp_col))
        p["u_amb"].value = tuple(map(float, amb))
        p["u_opacity"].value = float(opacity)
        p["u_texel"].value = (1.0 / self.w, 1.0 / self.h)
        p["u_bump"].value = float(bump)
        p["u_ink_gain"].value = float(ink_gain)
        self.vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)


class Canvas2:
    def __init__(self, w, h):
        self.cv = sk.Canvas(w, h)

    @property
    def c(self):
        return self.cv.c

    def clear(self):
        self.cv.clear()


# ------------------------------------------------------------ typewriter
def typewriter_line(c, text, x, y, f, reveal=1.0, seed=0, ink=1.0, rgb=(0, 0, 0)):
    """Monospace typed line: per-glyph baseline wobble, strike strength, slight rotation."""
    n = len(text)
    shown = n if reveal >= 1 else int(reveal * n)
    adv = f.getWidths(f.textToGlyphs("M"))[0]
    for i in range(shown):
        ch = text[i]
        if ch == " ":
            continue
        k = seed * 131 + i * 17
        dx = (hash1(k) - 0.5) * 1.1
        dy = (hash1(k + 1) - 0.5) * 1.6 + (0.6 if ch in "aeiou" else 0.0) * (hash1(k + 9) - 0.5)
        strike = 0.70 + 0.30 * hash1(k + 2)
        if hash1(k + 3) > 0.93:
            strike *= 0.72  # a weak key
        rot = (hash1(k + 4) - 0.5) * 1.2
        p = sk.paint(rgb, clamp(strike * ink))
        c.save()
        c.translate(x + i * adv + dx + adv / 2, y + dy)
        c.rotate(rot)
        blob = skia.TextBlob.MakeFromString(ch, f)
        c.drawTextBlob(blob, -adv / 2, 0, p)
        # double-strike ghost on some heavy keys (ribbon offset)
        if hash1(k + 5) > 0.85:
            q = sk.paint(rgb, strike * ink * 0.25)
            c.drawTextBlob(blob, -adv / 2 + 0.9, 0.5, q)
        c.restore()
    return shown


def stamp(c, lines, cx, cy, size, rot, rgb=(0.78, 0.10, 0.08), alpha=1.0, box=True, font_name="type_b",
          scale=1.0, tracking=0.12):
    """Rubber stamp: framed caps text, rotated. Ink starvation is applied in the paper shader."""
    c.save()
    c.translate(cx, cy)
    c.rotate(rot)
    c.scale(scale, scale)
    f = sk.font(font_name, size)
    widths = [sk.text_width(t, f, tracking) for t in lines]
    W_ = max(widths) + size * 0.9
    H_ = size * 1.18 * len(lines) + size * 0.5
    p = sk.paint(rgb, alpha)
    if box:
        c.drawRoundRect(skia.Rect.MakeXYWH(-W_ / 2, -H_ / 2, W_, H_), size * 0.12, size * 0.12,
                        sk.paint(rgb, alpha, stroke=size * 0.09))
        c.drawRoundRect(skia.Rect.MakeXYWH(-W_ / 2 + size * 0.16, -H_ / 2 + size * 0.16, W_ - size * 0.32,
                                           H_ - size * 0.32), size * 0.08, size * 0.08,
                        sk.paint(rgb, alpha * 0.9, stroke=size * 0.03))
    for i, t in enumerate(lines):
        y = -H_ / 2 + size * 0.25 + size * 1.18 * (i + 0.5) + size * 0.36
        sk.draw_text(c, t, 0, y, f, p, tracking=tracking, align="center")
    c.restore()
