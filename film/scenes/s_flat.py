"""S5 'You are Jack.' / S6 the days under one bulb + Nixon / S7 Lo Wu bridge + end card."""
import math
import numpy as np
import skia
from datetime import date
from engine import sk, ui
from engine.core import (W, H, INK, PAPER, PAPER_DIM, AMBER, RED, clamp, remap, ease, E_OUT, E_INOUT, E_SOFT,
                         lerp, hash1)
from scenes.base import Scene

D0 = date(1952, 11, 29)
D_FECTEAU = date(1971, 12, 13)
D_NIXON = date(1973, 1, 31)
D_DOWNEY = date(1973, 3, 12)
TOTAL = (D_DOWNEY - D0).days  # 7408


class Flat(Scene):
    """Skia-drawn full-frame scene."""

    def setup(self):
        self.cv = sk.Canvas(W, H)
        self.tex = None
        self.init()

    def init(self):
        pass

    def draw(self, t, fbo):
        fbo.use()
        fbo.clear(0, 0, 0, 1)
        c = self.cv.clear((0, 0, 0), 1.0).c
        self.paint(c, t)
        self.tex = self.gl.texture_rgba8(self.cv.flush(), tex=self.tex)
        self.gl.draw_plane(self.tex, np.diag([2, 2, 1, 1]), gain=self.gain(t))

    def gain(self, t):
        return 1.0


# ------------------------------------------------------------------ S5
class JackScene(Flat):
    name = "jack"

    def paint(self, c, t):
        c.clear(sk.color((0.008, 0.008, 0.010)))
        q = self.p.get("beats", {}).get("quote", 3.2)
        pre = ui.fade_in_out(t, 0.5, q - 0.1, 0.5, 0.4)
        ui.label(c, "AN OFFICER IN A LEATHER JACKET, WITH A PISTOL", W / 2, H / 2 + 10, 18, PAPER, 0.55 * pre,
                 align="center", tracking=0.28)
        a = clamp(remap(t, q, q + 0.35))
        flick = 0.85 + 0.15 * (hash1(int(t * 24)) > 0.12)
        f = sk.font("display_i", 132)
        sk.draw_text(c, "\u201cYou are Jack.\u201d", W / 2, H / 2 + 30, f, sk.paint(PAPER, a * flick), align="center")
        b = clamp(remap(t, q + 0.7, q + 1.2)) * (1 - clamp(remap(t, self.dur - 0.5, self.dur)))
        ui.label(c, "A CHINESE OFFICER, IN ENGLISH, AT THE MOMENT OF CAPTURE", W / 2, H / 2 + 120, 17, PAPER, 0.6 * b,
                 align="center", tracking=0.24)
        ui.source_tag(c, t, "DUJMOVIC, STUDIES IN INTELLIGENCE 50/4 (CIA, 2006)", 1.0, self.dur - 0.2)

    def gain(self, t):
        return clamp(remap(t, 0.0, 0.3)) * (1 - clamp(remap(t, self.dur - 0.35, self.dur)))

    def post(self, t):
        return dict(vignette=0.7, grain=0.07, bloom_str=0.12, halation=0.1, exposure=1.0)


# ------------------------------------------------------------------ S6
EVENTS = [  # date, english, chinese(optional), side
    (date(1953, 12, 4), "PRESUMED DEAD BY THE CIA, LETTERS TO FAMILIES", ""),
    (date(1954, 11, 23), "BEIJING ANNOUNCES VERDICTS: LIFE \u00b7 20 YEARS", "\u5ba1\u5224"),
    (date(1958, 1, 15), "DOWNEY'S MOTHER'S FIRST VISIT", ""),
    (date(1971, 12, 13), "FECTEAU FREED AT LO WU \u00b7 6,953 DAYS", ""),
]


class DaysScene(Flat):
    name = "days"

    def init(self):
        self.b = dict(sweep0=0.3, sweep1=6.0, nixon=6.3, hl0=7.0)
        self.b.update(self.p.get("beats", {}))
        self.years = list(range(1952, 1974))
        self.x0, self.y0 = 250, 175
        self.cell, self.row = 3.0, 34.0

    def day_xy(self, d):
        y = d.year
        doy = (d - date(y, 1, 1)).days
        return self.x0 + doy * self.cell, self.y0 + (y - 1952) * self.row

    def paint(self, c, t):
        b = self.b
        c.clear(sk.color((0.012, 0.012, 0.014)))
        # the single bulb that burned day and night
        bx, by = W / 2, -40
        glow = 0.9 + 0.1 * math.sin(t * 37) * (hash1(int(t * 24)) > 0.7)
        for r, a in ((900, 0.05), (520, 0.06), (260, 0.08)):
            p = skia.Paint(AntiAlias=True)
            p.setShader(skia.GradientShader.MakeRadial(skia.Point(bx, by), r,
                        [sk.color((1.0, 0.82, 0.55), a * glow), sk.color((1.0, 0.8, 0.5), 0.0)]))
            c.drawRect(skia.Rect.MakeWH(W, H), p)
        k = clamp(remap(t, b["sweep0"], b["sweep1"]))
        k = ease(k, E_INOUT)
        now = D0.toordinal() + int((D_NIXON - D0).days * k)
        nowd = date.fromordinal(now)
        fill_d = sk.paint(PAPER, 0.85)
        fill_f = sk.paint(AMBER, 0.85)
        empty = sk.paint(PAPER, 0.07)
        ys_a = clamp(remap(t, 0.0, 0.6))
        for yi, y in enumerate(self.years):
            yy = self.y0 + yi * self.row
            ui.label(c, str(y), self.x0 - 24, yy + 10, 15, PAPER, 0.55 * ys_a, align="right", tracking=0.1)
            ndays = (date(y + 1, 1, 1) - date(y, 1, 1)).days
            c.drawRect(skia.Rect.MakeXYWH(self.x0, yy, ndays * self.cell, 6), empty)
            c.drawRect(skia.Rect.MakeXYWH(self.x0, yy + 9, ndays * self.cell, 3), empty)
            start = max(date(y, 1, 1), D0)
            end_d = min(date(y + 1, 1, 1), nowd, D_DOWNEY)
            if end_d > start:
                x0 = self.x0 + (start - date(y, 1, 1)).days * self.cell
                w = (end_d - start).days * self.cell
                c.drawRect(skia.Rect.MakeXYWH(x0, yy, w, 6), fill_d)
                end_f = min(end_d, D_FECTEAU)
                if end_f > start:
                    c.drawRect(skia.Rect.MakeXYWH(x0, yy + 9, (end_f - start).days * self.cell, 3), fill_f)
        # legend
        la = clamp(remap(t, 0.2, 0.8))
        ui.label(c, "EACH MARK IS ONE DAY IN CHINESE CUSTODY", self.x0, self.y0 - 62, 16, PAPER, 0.75 * la, tracking=0.22)
        c.drawRect(skia.Rect.MakeXYWH(self.x0, self.y0 - 38, 24, 6), sk.paint(PAPER, 0.85 * la))
        ui.label(c, "DOWNEY", self.x0 + 34, self.y0 - 30, 14, PAPER, 0.75 * la, tracking=0.2)
        c.drawRect(skia.Rect.MakeXYWH(self.x0 + 140, self.y0 - 37, 24, 3), sk.paint(AMBER, 0.85 * la))
        ui.label(c, "FECTEAU", self.x0 + 174, self.y0 - 30, 14, AMBER, 0.75 * la, tracking=0.2)
        # cursor + counter
        if 0 < k < 1:
            cx, cy = self.day_xy(nowd)
            c.drawLine(cx, cy - 6, cx, cy + 16, sk.paint((1, 0.9, 0.7), 0.95, stroke=2))
        days = now - D0.toordinal()
        ui.label(c, f"DAY {days:,}", W - 90, self.y0 - 30, 26, PAPER, 0.9 * la, align="right", font_name="mono_m",
                 tracking=0.12)
        ui.label(c, nowd.strftime("%d %b %Y").upper(), W - 90, self.y0 - 2, 15, PAPER, 0.55 * la, align="right")
        # events
        for d, txt, zh in EVENTS:
            if nowd >= d:
                ea = clamp((now - d.toordinal()) / 120.0)
                x, y = self.day_xy(d)
                c.drawCircle(x, y + 3, 4, sk.paint(AMBER if "FECTEAU" in txt else PAPER, ea))
                ex_ = self.x0 + 366 * self.cell + 30
                c.drawLine(x + 5, y + 3, ex_ - 8, y + 3, sk.paint(PAPER, 0.25 * ea, stroke=1.0))
                ui.label(c, d.strftime("%b %Y").upper(), ex_, y + 2, 13, AMBER if "FECTEAU" in txt else PAPER, 0.7 * ea, tracking=0.16)
                ui.label(c, txt, ex_, y + 22, 14, PAPER, 0.9 * ea, tracking=0.1)
        # Washington's line, held until the Nixon admission
        wa = clamp(remap(t, b["sweep0"] + 0.6, b["sweep0"] + 1.2))
        x, y = 72, 980
        ui.label(c, "WASHINGTON'S PUBLIC LINE, 1952\u20131973", x, y - 34, 14, PAPER, 0.5 * wa, tracking=0.24)
        ui.label(c, "\u201cDEPARTMENT OF THE ARMY CIVILIANS\u201d", x, y, 24, PAPER, 0.85 * wa, font_name="cond_m",
                 tracking=0.08)
        st = clamp(remap(t, b["hl0"] + 0.4, b["hl0"] + 0.8))
        if st > 0:
            c.drawLine(x - 6, y - 8, x - 6 + 520 * ease(st, E_OUT), y - 8, sk.paint(RED, 0.95, stroke=3))
        # Nixon transcript card
        na = clamp(remap(t, b["nixon"], b["nixon"] + 0.5))
        if na > 0:
            cx, cy = 1000, 760
            off = (1 - ease(na, E_OUT)) * 30
            card = skia.Rect.MakeXYWH(cx, cy + off, 860, 250)
            c.drawRect(card, sk.paint((0.88, 0.85, 0.78), 0.97 * na))
            ui.label(c, "THE PRESIDENT'S NEWS CONFERENCE  \u00b7  31 JANUARY 1973", cx + 36, cy + 52 + off, 15, (0.15, 0.14, 0.13),
                     na, tracking=0.16)
            f = sk.font("type", 31)
            sk.draw_text(c, "Q. \"Downey, also?\"", cx + 36, cy + 108 + off, f, sk.paint((0.12, 0.11, 0.1), na))
            sk.draw_text(c, "THE PRESIDENT. \"Downey is a different case,", cx + 36, cy + 152 + off, f,
                         sk.paint((0.12, 0.11, 0.1), na))
            hl = ease(clamp(remap(t, b["hl0"], b["hl0"] + 0.8)), E_OUT)
            line2 = "as you know. Downey involves a CIA agent.\""
            w2 = sk.text_width("as you know. ", f)
            w3 = sk.text_width("Downey involves a CIA agent.", f)
            c.drawRect(skia.Rect.MakeXYWH(cx + 36 + w2 - 4, cy + 172 + off, (w3 + 8) * hl, 38),
                       sk.paint((1.0, 0.78, 0.25), 0.75 * na))
            sk.draw_text(c, line2, cx + 36, cy + 198 + off, f, sk.paint((0.12, 0.11, 0.1), na))
            ui.label(c, "AMERICAN PRESIDENCY PROJECT TRANSCRIPT", cx + 36, cy + 236 + off, 12, (0.3, 0.28, 0.25), na * 0.8,
                     tracking=0.2)
        ui.source_tag(c, t, "DUJMOVIC (CIA, 2006) \u00b7 FRUS 1952\u201354 XIV DOC 406 \u00b7 AP, 13 DEC 1971", 0.6, self.dur - 0.2)

    def gain(self, t):
        return clamp(remap(t, 0.0, 0.4))

    def post(self, t):
        return dict(vignette=0.55, grain=0.05, bloom_str=0.08, halation=0.06)


# ------------------------------------------------------------------ S7
class BridgeScene(Flat):
    name = "bridge"

    def init(self):
        self.b = dict(walk0=0.2, border=4.6, final=6.0, card=9.0)
        self.b.update(self.p.get("beats", {}))
        r = np.random.default_rng(5)
        self.mist = [(r.uniform(-400, W + 400), r.uniform(560, 760), r.uniform(300, 700), r.uniform(0.04, 0.09),
                      r.uniform(6, 18)) for _ in range(14)]
        self.hills = []
        for layer in range(3):
            xs = np.linspace(-100, W + 100, 60)
            base = 560 - layer * 40
            ys = base - 60 * (0.5 + 0.5 * np.sin(xs * 0.004 + layer * 2)) - 30 * np.sin(xs * 0.011 + layer)
            self.hills.append((xs, ys, layer))

    def paint(self, c, t):
        b = self.b
        drift = t * 6.0
        sky = skia.Paint()
        sky.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                      [sk.color((0.10, 0.12, 0.15)), sk.color((0.42, 0.44, 0.45)), sk.color((0.30, 0.32, 0.34))],
                      [0.0, 0.62, 1.0]))
        c.drawRect(skia.Rect.MakeWH(W, H), sky)
        sun = skia.Paint(AntiAlias=True)
        sun.setShader(skia.GradientShader.MakeRadial(skia.Point(1350, 470), 520,
                      [sk.color((1.0, 0.86, 0.66), 0.35), sk.color((1.0, 0.8, 0.6), 0.0)]))
        c.drawRect(skia.Rect.MakeWH(W, H), sun)
        for xs, ys, layer in self.hills:
            pth = sk.polyline(np.c_[xs - drift * (0.2 + layer * 0.15), ys])
            pth.lineTo(W + 200, H)
            pth.lineTo(-200, H)
            pth.close()
            v = 0.30 - layer * 0.05
            c.drawPath(pth, sk.paint((v * 0.9, v * 0.95, v), 0.9))
        # river
        c.drawRect(skia.Rect.MakeXYWH(0, 700, W, H - 700), sk.paint((0.20, 0.22, 0.24), 1.0))
        for i in range(40):
            y = 708 + i * 9 + (i * i) * 0.3
            x = (hash1(i) * W + drift * 4 * (1 + hash1(i + 3))) % (W + 300) - 150
            c.drawLine(x, y, x + 60 + 120 * hash1(i * 2), y, sk.paint((0.6, 0.62, 0.62), 0.10))
        # bridge (side elevation): deck, Pratt truss, piers
        deck_y, x_l, x_r = 640, -60, W + 60
        dark = (0.07, 0.075, 0.08)
        ox = -drift * 1.2
        c.drawRect(skia.Rect.MakeXYWH(x_l, deck_y, x_r - x_l, 14), sk.paint(dark, 1))
        span = 240
        for k in range(-1, 10):
            x = k * span + ox
            c.drawRect(skia.Rect.MakeXYWH(x - 12, deck_y + 14, 24, 80), sk.paint(dark, 1))
            p = sk.paint(dark, 1, stroke=5)
            c.drawLine(x, deck_y, x, deck_y - 120, p)
            c.drawLine(x, deck_y - 120, x + span, deck_y - 120, p)
            c.drawLine(x, deck_y - 120, x + span / 2, deck_y, sk.paint(dark, 1, stroke=3.5))
            c.drawLine(x + span, deck_y - 120, x + span / 2, deck_y, sk.paint(dark, 1, stroke=3.5))
            c.drawLine(x + span / 2, deck_y, x + span / 2, deck_y - 120, sk.paint(dark, 1, stroke=2.5))
        # border marker mid-bridge
        bx = 1060 + ox
        c.drawLine(bx, deck_y - 170, bx, deck_y, sk.paint(PAPER, 0.5, stroke=1.2))
        ui.label(c, "BORDER", bx, deck_y - 182, 13, PAPER, 0.6, align="center", tracking=0.4)
        ui.cjk(c, "\u7f57\u6e56", bx - 260, deck_y - 230, 26, PAPER, 0.55, align="center")
        ui.label(c, "LO WU", bx - 260, deck_y - 196, 15, PAPER, 0.7, align="center", tracking=0.4)
        ui.label(c, "CHINA", bx - 120, deck_y + 48, 14, PAPER, 0.45, align="center", tracking=0.5)
        ui.label(c, "HONG KONG", bx + 140, deck_y + 48, 14, PAPER, 0.45, align="center", tracking=0.5)
        # the walker
        wk = clamp(remap(t, b["walk0"], b["final"] + 1.5))
        wx = lerp(560, 1340, wk) + ox
        step = math.sin(t * 7.5)
        h = 34
        fy = deck_y
        fig = sk.paint((0.03, 0.03, 0.035), 1)
        c.drawCircle(wx, fy - h + 3.5, 3.4, fig)
        c.drawRoundRect(skia.Rect.MakeXYWH(wx - 4, fy - h + 7, 8, 15), 2, 2, fig)
        lp = sk.paint((0.03, 0.03, 0.035), 1, stroke=2.4)
        c.drawLine(wx, fy - h + 21, wx + step * 5, fy, lp)
        c.drawLine(wx, fy - h + 21, wx - step * 5, fy, lp)
        # mist layers in front
        for (mx, my, mw, ma, ms) in self.mist:
            x = (mx + t * ms) % (W + 800) - 400
            p = skia.Paint(AntiAlias=True)
            p.setShader(skia.GradientShader.MakeRadial(skia.Point(x, my), mw,
                        [sk.color((0.75, 0.76, 0.76), ma), sk.color((0.75, 0.76, 0.76), 0.0)]))
            c.drawRect(skia.Rect.MakeWH(W, H), p)
        # captions
        ui.caption(c, t, "LO WU BRIDGE, HONG KONG BORDER  \u00b7  12 MARCH 1973", 72, 120, 0.4, b["final"], size=18)
        ui.caption(c, t, "A BRITISH SOLDIER SALUTED HIM", 72, 150, 1.4, b["final"], size=15)
        # day counter -> final number
        k = clamp(remap(t, 0.0, b["border"]))
        days = int(round(lerp((D_NIXON - D0).days, TOTAL, ease(k, E_OUT))))
        fa = clamp(remap(t, b["final"] - 0.2, b["final"] + 0.6))
        ca = 1 - fa
        if ca > 0:
            ui.label(c, f"DAY {days:,}", W - 120, 120, 26, PAPER, 0.9 * ca, align="right", font_name="mono_m", tracking=0.12)
        if fa > 0:
            veil = sk.paint((0.02, 0.022, 0.026), 0.78 * fa)
            c.drawRect(skia.Rect.MakeWH(W, H), veil)
            f = sk.font("display", 230)
            sc = 1.0 + 0.03 * (1 - ease(fa, E_OUT))
            c.save()
            c.translate(W / 2, H / 2 + 40)
            c.scale(sc, sc)
            sk.draw_text(c, f"{TOTAL:,}", 0, 0, f, sk.paint(PAPER, fa), align="center")
            c.restore()
            ui.label(c, "DAYS", W / 2, H / 2 + 120, 22, PAPER, 0.8 * fa, align="center", tracking=0.8)
            ui.label(c, "29 NOV 1952  \u2014  12 MAR 1973", W / 2, H / 2 + 162, 15, PAPER, 0.5 * fa, align="center",
                     tracking=0.3)
        # end card
        ea = clamp(remap(t, b["card"], b["card"] + 0.6))
        if ea > 0:
            c.drawRect(skia.Rect.MakeWH(W, H), sk.paint((0.01, 0.01, 0.012), ea))
            sk.draw_text(c, "The Pickup", W / 2, H / 2 - 10, sk.font("display", 120), sk.paint(PAPER, ea), align="center")
            ui.label(c, "CIA AIR PICKUP  \u00b7  JILIN, CHINA  \u00b7  29 NOVEMBER 1952", W / 2, H / 2 + 60, 17, PAPER, 0.7 * ea,
                     align="center", tracking=0.3)
            ui.label(c, "SOURCES: N. DUJMOVIC, STUDIES IN INTELLIGENCE 50/4 (2006) \u00b7 FRUS \u00b7 DPAA \u00b7 AMERICAN PRESIDENCY PROJECT",
                     W / 2, H - 90, 12, PAPER, 0.45 * ea, align="center", tracking=0.18)

    def gain(self, t):
        return clamp(remap(t, 0.0, 0.5)) * (1 - clamp(remap(t, self.dur - 0.6, self.dur)))

    def post(self, t):
        b = self.b
        fa = clamp(remap(t, b["final"] - 0.2, b["final"] + 0.6))
        return dict(vignette=0.5, grain=0.05, bloom_str=0.08, halation=0.05, letterbox=0.115 * (1 - fa), sat=0.7)
