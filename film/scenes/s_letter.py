"""S1 - The condolence letter (verbatim text quoted in Dujmovic, CIA 2006; layout reconstructed)."""
import math
import numpy as np
from engine import sk, ui, docs
from engine.core import (W, H, INK, PAPER, RED, clamp, remap, ease, E_INOUT, E_SOFT, E_OUT, lerp, Track,
                         look_at, perspective, scale, translate, rot_z)
from scenes.base import Scene

PW, PH = 2048, 2650
BODY = [
    "Dear Mrs. Downey:",
    "",
    "I have learned that your son was a passenger on",
    "a commercial plane flight between South Korea and",
    "Japan which is now overdue and that there is grave",
    "fear that he may have been lost.",
]


class LetterScene(Scene):
    name = "letter"

    def setup(self):
        self.pg = docs.PaperGL(self.gl, PW, PH, seed=11)
        self.f_type = sk.font("type", 50)
        self.f_head = sk.font("type_b", 44)
        self.b = dict(type0=0.9, type1=6.2, strike=99.0)
        self.b.update(self.p.get("beats", {}))
        self.x0, self.y0, self.lh = 250, 1000, 84
        self.chars = sum(len(l) for l in BODY)
        self.last_n = -1
        self.adv = self.f_type.getWidths(self.f_type.textToGlyphs("M"))[0]

    def uv_to_world(self, u, v):
        return np.array([(u / PW - 0.5) * 0.85, (0.5 - v / PH) * 1.10, 0.0])

    def typed_pos(self, n):
        """Pixel position of the n-th typed character (the 'strike point')."""
        acc = 0
        for i, l in enumerate(BODY):
            if n <= acc + len(l):
                return self.x0 + (n - acc) * self.adv, self.y0 + i * self.lh
            acc += len(l)
        return self.x0 + len(BODY[-1]) * self.adv, self.y0 + (len(BODY) - 1) * self.lh

    def redraw(self, n):
        ink = self.pg.ink
        ink.clear()
        c = ink.c
        docs.typewriter_line(c, "CENTRAL INTELLIGENCE AGENCY", 1024 - 27 * 13.5, 420, self.f_head, seed=91)
        docs.typewriter_line(c, "WASHINGTON 25, D. C.", 1024 - 20 * 15, 486, self.f_type, seed=92)
        docs.typewriter_line(c, "OFFICE OF THE DIRECTOR", 1024 - 22 * 15, 552, self.f_type, seed=93, ink=0.85)
        docs.typewriter_line(c, "December 1952", 1400, 760, self.f_type, seed=94)
        acc = 0
        for i, l in enumerate(BODY):
            k = clamp((n - acc) / max(len(l), 1))
            docs.typewriter_line(c, l, self.x0, self.y0 + i * self.lh, self.f_type, reveal=k if k < 1 else 1.0, seed=i + 3)
            acc += len(l)
        self.pg.red.clear()

    def draw(self, t, fbo):
        b = self.b
        fbo.use()
        fbo.clear(0.006, 0.006, 0.008, 1.0)
        n = int(self.chars * clamp((t - b["type0"]) / (b["type1"] - b["type0"])))
        if n != self.last_n:
            self.redraw(n)
            self.pg.upload()
            self.last_n = n
        sx, sy = self.typed_pos(n)
        strike = self.uv_to_world(sx, sy)
        # camera: macro tracking the strike point, then a slow pull back to reveal the page
        pull = ease(remap(t, b["type1"] - 2.4, b["type1"] + 0.8), E_INOUT)
        focus = lerp(strike + np.array([0.0, -0.01, 0]), np.array([0.02, 0.10, 0.0]), pull)
        dist = lerp(0.20, 0.92, pull)
        eye = focus + np.array([0.05 * (1 - pull) + 0.03 * math.sin(t * 0.3), -0.10 * (1 - pull) - 0.18 * pull, dist])
        vp = perspective(lerp(30, 34, pull), W / H, 0.01, 20) @ look_at(eye, focus, (0, 1, 0))
        model = rot_z(0.012) @ scale(0.85, 1.10, 1)
        lamp_in = clamp(remap(t, 0.0, 1.4))
        lamp = np.array([-0.45 + 0.05 * math.sin(t * 0.5), 0.55, 0.42])
        self.pg.draw(vp @ model, model, lamp=lamp, lamp_col=tuple(np.array([2.4, 2.05, 1.6]) * lamp_in),
                     amb=(0.035 * lamp_in, 0.037 * lamp_in, 0.045 * lamp_in), bump=0.9)
        self.ov = getattr(self, "ov", sk.Canvas(W, H))
        c = self.ov.clear().c
        ui.caption(c, t, "TEXT AS QUOTED BY CIA HISTORIAN N. DUJMOVIC (2006)  \u00b7  LAYOUT RECONSTRUCTED", 72, H - 72,
                   b["type1"] - 1.0, self.dur, size=13)
        self.ovt = self.gl.texture_rgba8(self.ov.flush(), tex=getattr(self, "ovt", None))
        self.gl.draw_plane(self.ovt, np.diag([2, 2, 1, 1]))

    def post(self, t):
        return dict(exposure=1.0, vignette=0.62, grain=0.05, bloom_str=0.05, halation=0.05,
                    fade=1 - clamp(remap(t, 0.0, 0.8)), ca=0.002)
