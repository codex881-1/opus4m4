"""S2 - Map: the cover story route vs. the real route into Jilin."""
import math
import os
import numpy as np
import skia
from PIL import Image
from scipy import ndimage
from engine import sk, geo, ui
from engine.core import (W, H, INK, PAPER, AMBER, RED, Track, E_INOUT, E_OUT, E_SOFT, ease, remap, clamp,
                         look_at, perspective, project, scale, lerp)
from engine import c47
from scenes.base import Scene

K16 = (127.1139, 37.4459)          # Seoul Air Base (K-16), Seongnam
SITE = (128.4945, 42.4852)         # DPAA excavation datum N42 29' 06.6" E128 29' 40.3"
TOKYO = (139.75, 35.69)
MUKDEN = (123.43, 41.80)
PLACES = {  # label, chinese, lon, lat, align
    "SEOUL": ("서울", 126.98, 37.57, "right"),
    "PYONGYANG": ("평양", 125.75, 39.02, "right"),
    "MUKDEN": ("沈阳", 123.43, 41.80, "right"),
    "JILIN": ("吉林", 126.55, 43.85, "right"),
    "TOKYO": ("東京", 139.75, 35.69, "left"),
}


def build_basemap(proj):
    f = os.path.join(geo.CACHE, f"basemap2_{proj.width}.png")
    if os.path.exists(f):
        return np.asarray(Image.open(f))
    e = geo.relief_raster(proj)
    h, w = e.shape
    cell = 1 / proj.s
    hs = geo.hillshade(e, cell, zf=5.0)
    hs_sea = geo.hillshade(np.minimum(e, 0), cell, zf=2.0)
    # land mask from Natural Earth polygons
    cv = sk.Canvas(w, h).clear()
    pa = sk.paint((1, 1, 1), 1.0)
    path = skia.Path()
    path.setFillType(skia.PathFillType.kEvenOdd)
    for pts in geo.load_rings("ne_10m_land", proj):
        path.addPath(sk.polyline(pts, closed=True))
    cv.c.drawPath(path, pa)
    land = cv.flush()[..., 3].astype(np.float32) / 255.0
    land = ndimage.gaussian_filter(land, 0.7)
    elev_n = np.clip(e / 2200.0, 0, 1)
    ao = np.clip((ndimage.gaussian_filter(e, 6) - e) / 300.0, -1, 1)
    base = np.array([0.105, 0.102, 0.098])
    lum = (0.42 + 1.05 * hs) * (1 - 0.18 * np.clip(ao, 0, 1)) * (1 + 0.35 * elev_n)
    land_rgb = base[None, None] * lum[..., None]
    land_rgb += np.array([0.02, 0.025, 0.035])[None, None] * elev_n[..., None]  # cool high ground (snow line hint)
    depth = np.clip(-e / 4000.0, 0, 1)
    sea_rgb = (np.array([0.020, 0.028, 0.038])[None, None] * (1 - 0.55 * depth[..., None])
               * (0.8 + 0.4 * hs_sea[..., None]))
    rgb = sea_rgb * (1 - land[..., None]) + land_rgb * land[..., None]
    img = np.clip(rgb * 255 * 1.0, 0, 255).astype(np.uint8)
    img = np.dstack([img, np.full((h, w), 255, np.uint8)])
    out = img
    Image.fromarray(out).save(f)
    return out


def c47_planform(span_px):
    """Top-view outline of the C-47 from the same parametric model (x forward)."""
    s = np.linspace(0, 19.43, 40)
    wf = np.array([c47._interp_fus(x)[0] for x in s])
    fus_r = [(c47.S0 - x, w) for x, w in zip(s, wf)]
    fus_l = [(c47.S0 - x, -w) for x, w in zip(s[::-1], wf[::-1])]
    le, te = 6.05, 10.35
    wing_r = [(c47.S0 - le, 1.2), (c47.S0 - (te - 1.55), 14.4), (c47.S0 - te, 14.4), (c47.S0 - te, 1.2)]
    tail_r = [(c47.S0 - 16.5, 0.5), (c47.S0 - 17.9, 4.0), (c47.S0 - 19.2, 4.0), (c47.S0 - 19.2, 0.5)]
    k = span_px / 28.96
    polys = []
    for poly in (fus_r + fus_l, wing_r, [(x, -y) for x, y in wing_r], tail_r, [(x, -y) for x, y in tail_r]):
        polys.append([(x * k, y * k) for x, y in poly])
    return polys


class MapScene(Scene):
    name = "map"

    def setup(self):
        self.proj = geo.MapProj(width=4096)
        base = build_basemap(self.proj)
        self.tex = self.gl.texture_rgba8(base, mipmap=True)
        self.mw, self.mh = self.proj.width / 1000, self.proj.height / 1000
        self.ov = sk.Canvas(W, H)
        self.ovtex = None
        P = lambda ll: self.proj.px(*ll)[0]
        self.k16, self.site, self.tokyo = P(K16), P(SITE), P(TOKYO)
        # real route: gentle arc K-16 -> site (schematic)
        a, b = self.k16, self.site
        mid = (a + b) / 2 + np.array([-(b - a)[1], (b - a)[0]]) * 0.10
        us = np.linspace(0, 1, 80)
        self.route_pts = np.array([(1 - u) ** 2 * a + 2 * (1 - u) * u * mid + u ** 2 * b for u in us])
        c, d = self.k16, self.tokyo
        mid2 = (c + d) / 2 + np.array([(d - c)[1], -(d - c)[0]]) * -0.12
        self.cover_pts = np.array([(1 - u) ** 2 * c + 2 * (1 - u) * u * mid2 + u ** 2 * d for u in us])
        self.places = {k: (v[0], P((v[1], v[2])), v[3]) for k, v in PLACES.items()}
        self.planform = c47_planform(46)
        self.vec = []
        for pts, pr in geo.load_lines("ne_10m_coastline", self.proj):
            self.vec.append((geo.simplify(pts, 1.2), PAPER, 0.50, 1.3, None))
        intl = lambda p: p.get("FEATURECLA", "").startswith("International")
        for pts, pr in geo.load_lines("ne_10m_admin_0_boundary_lines_land", self.proj, filt=intl):
            self.vec.append((geo.simplify(pts, 1.2), PAPER, 0.40, 1.5, (7, 5)))
        riv = lambda p: (p.get("scalerank") or 9) <= 6
        for pts, pr in geo.load_lines("ne_10m_rivers_lake_centerlines", self.proj, filt=riv):
            self.vec.append((geo.simplify(pts, 1.5), (0.45, 0.55, 0.65), 0.28, 1.0, None))
        self._vec_all = np.concatenate([v[0] for v in self.vec])
        self._vec_split = np.cumsum([len(v[0]) for v in self.vec])[:-1]
        b = self.p.get("beats", {})
        self.b = dict(cover=0.0, strike=2.2, real=3.6, cards=4.4, zoom=8.6, site=9.6)
        self.b.update(b)
        bb = self.b
        cx = self.k16 * 0.58 + self.tokyo * 0.42
        self.cam_target = Track([
            (0.0, tuple(cx + [0, 40]), E_INOUT),
            (bb["real"], tuple(cx + [0, 40]), E_INOUT),
            (bb["real"] + 3.4, tuple((self.k16 + self.site) / 2 + [60, 0]), E_INOUT),
            (bb["zoom"], tuple((self.k16 + self.site) / 2 + [60, 0]), E_INOUT),
            (bb["zoom"] + 2.6, tuple(self.site + [0, 10]), E_INOUT),
        ])
        self.cam_dist = Track([(0.0, 2.15, E_INOUT), (bb["real"], 2.05, E_INOUT), (bb["real"] + 3.4, 2.25, E_INOUT),
                               (bb["zoom"], 2.15, E_INOUT), (bb["zoom"] + 2.6, 0.55, E_INOUT)])
        self.cam_tilt = Track([(0.0, 8.0, E_INOUT), (bb["real"] + 3.4, 24.0, E_INOUT), (bb["zoom"], 26.0, E_INOUT),
                               (bb["zoom"] + 2.6, 44.0, E_INOUT)])

    # map px -> world
    def w(self, uv):
        uv = np.atleast_2d(uv)
        return np.c_[(uv[:, 0] / self.proj.width - 0.5) * self.mw, (0.5 - uv[:, 1] / self.proj.height) * self.mh,
                     np.zeros(len(uv))]

    def camera(self, t):
        tgt = np.array(self.cam_target(t))
        tw = self.w(tgt)[0]
        d, tilt = self.cam_dist(t), math.radians(self.cam_tilt(t))
        drift = np.array([math.sin(t * 0.21) * 0.01, math.cos(t * 0.17) * 0.008, 0])
        eye = tw + np.array([0, -math.sin(tilt) * d, math.cos(tilt) * d]) + drift
        vp = perspective(30, W / H, 0.01, 50) @ look_at(eye, tw + drift * 0.5, (0, 1, 0))
        return vp, eye

    def scr(self, vp, uv):
        s, depth, vis = project(self.w(uv), vp)
        return s

    def runs(self, vp, uv):
        """Visible screen-space runs (drops points behind the camera)."""
        s, depth, vis = project(self.w(uv), vp)
        vis &= depth > 0.02
        if vis.all():
            return [s]
        idx = np.where(vis)[0]
        if len(idx) < 2:
            return []
        return [s[r] for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1) if len(r) > 1]

    def draw(self, t, fbo):
        gl = self.gl
        fbo.use()
        fbo.clear(*INK, 1.0)
        vp, eye = self.camera(t)
        model = scale(self.mw, self.mh, 1)
        gl.draw_plane(self.tex, vp @ model, gain=1.0)
        self.draw_overlay(t, vp)
        self.ovtex = gl.texture_rgba8(self.ov.flush(), tex=self.ovtex)
        gl.draw_plane(self.ovtex, scale(2, 2, 1), gain=1.0)

    def draw_overlay(self, t, vp):
        b = self.b
        cv = self.ov.clear()
        c = cv.c
        # graticule ticks (projected)
        for lon in range(118, 144, 2):
            pts = self.proj.px(np.full(40, lon), np.linspace(31, 48.5, 40))
            for r in self.runs(vp, pts):
                c.drawPath(sk.polyline(r), sk.paint(PAPER, 0.045, stroke=1))
        for lat in range(32, 49, 2):
            pts = self.proj.px(np.linspace(117, 143, 60), np.full(60, lat))
            for r in self.runs(vp, pts):
                c.drawPath(sk.polyline(r), sk.paint(PAPER, 0.045, stroke=1))
        allp, dep, vis = project(self.w(self._vec_all), vp)
        ok = vis & (dep > 0.02)
        for (pts, rgb, al, wd, dash), sp, okp in zip(self.vec, np.split(allp, self._vec_split), np.split(ok, self._vec_split)):
            pa = sk.paint(rgb, al, stroke=wd)
            if dash:
                sk.dashed(pa, *dash)
            if okp.all():
                c.drawPath(sk.polyline(sp), pa)
            else:
                idx = np.where(okp)[0]
                for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                    if len(r) > 1:
                        c.drawPath(sk.polyline(sp[r]), pa)
        # place labels
        for name, (zh, uv, al) in self.places.items():
            s = self.scr(vp, uv)[0]
            a = 0.75 * clamp(remap(t, 0.3, 1.2))
            if name == "TOKYO":
                a *= 1 - clamp(remap(t, b["real"] + 1, b["real"] + 2))
            c.drawCircle(s[0], s[1], 3.2, sk.paint(PAPER, a))
            dx = -12 if al == "right" else 12
            ui.label(c, name, s[0] + dx, s[1] + 6, 18, PAPER, a, align=al)
            ui.cjk(c, zh, s[0] + dx, s[1] + 31, 18, PAPER, a * 0.55, align=al)
        # country names (large tracked caps)
        for txt, zh, ll in (("CHINA", "中国", (123.2, 44.9)), ("KOREA", "", (127.6, 38.6)), ("JAPAN", "", (138.2, 37.2)),
                            ("SEA OF JAPAN", "", (133.5, 40.0))):
            s = self.scr(vp, self.proj.px(*ll))[0]
            a = 0.22 * clamp(remap(t, 0.5, 1.5))
            ui.label(c, txt, s[0], s[1], 22 if txt != "SEA OF JAPAN" else 15, PAPER, a, align="center", tracking=0.5,
                     font_name="cond")
            if zh:
                ui.cjk(c, zh, s[0], s[1] + 34, 22, PAPER, a * 0.9, align="center", tracking=0.6)
        # --- cover-story route (dashed) and its strike-out
        ck = ease(remap(t, b["cover"] + 0.2, b["cover"] + 1.6), E_SOFT)
        fade = 1 - clamp(remap(t, b["real"] + 0.6, b["real"] + 1.8))
        if ck > 0 and fade > 0:
            path = sk.smooth_path(self.scr(vp, self.cover_pts))
            sub = sk.sub_path(path, 0, ck)
            strike = clamp(remap(t, b["strike"], b["strike"] + 0.35))
            col = tuple(lerp(a_, b_, strike) for a_, b_ in zip(PAPER, RED))
            c.drawPath(sub, sk.dashed(sk.paint(col, 0.85 * fade, stroke=2.6), 12, 9, -t * 30))
            (mx, my), _ = sk.point_at(path, 0.55)
            ui.label(c, "OFFICIAL ACCOUNT", mx + 18, my - 40, 19, PAPER, 0.9 * fade * ck, tracking=0.22)
            ui.label(c, "CAT FLIGHT  KOREA \u2192 JAPAN", mx + 18, my - 14, 18, PAPER, 0.65 * fade * ck)
            ui.label(c, "\"LOST AT SEA\"", mx + 18, my + 12, 18, PAPER, 0.65 * fade * ck)
            if strike > 0:
                # hand-ruled strike: two red strokes through the label block
                p = sk.paint(RED, 0.95 * fade, stroke=3.0)
                L = 300 * ease(strike, E_OUT)
                c.drawLine(mx + 10, my - 6, mx + 10 + L, my - 2, p)
                ui.label(c, "COVER STORY", mx + 18, my + 50, 21, RED, fade * strike, tracking=0.3, font_name="mono_sb")
        # --- the real route
        rk = ease(remap(t, b["real"], b["real"] + 3.2), E_INOUT)
        if rk > 0:
            spts = self.scr(vp, self.route_pts)
            path = sk.smooth_path(spts)
            sub = sk.sub_path(path, 0, rk)
            sk.glow_stroke(c, sub, AMBER, 2.6, 0.95, glow=9, glow_a=0.45)
            (px, py), (tx, ty) = sk.point_at(path, rk)
            ang = math.degrees(math.atan2(ty, tx))
            c.save()
            c.translate(px, py)
            c.rotate(ang)
            for poly in self.planform:
                c.drawPath(sk.polyline(poly, closed=True), sk.paint(AMBER, 0.95))
            c.restore()
            k = self.scr(vp, self.k16)[0]
            ui.label(c, "K-16  21:40", k[0] + 14, k[1] + 26, 18, AMBER, 0.9 * clamp(remap(t, b["real"], b["real"] + .5)))
            if rk > 0.25:
                (lx, ly), _ = sk.point_at(path, 0.45)
                a = clamp(remap(rk, 0.25, 0.45))
                ui.label(c, "UNMARKED C-47 \u00b7 CIVIL AIR TRANSPORT (CIA)", lx + 26, ly, 18, AMBER, 0.9 * a)
                ui.label(c, "29 NOV 1952 \u00b7 NIGHT", lx + 26, ly + 26, 17, AMBER, 0.6 * a)
        # --- crew cards
        ca = clamp(remap(t, b["cards"], b["cards"] + 0.5)) * (1 - clamp(remap(t, b["zoom"] - 0.3, b["zoom"] + 0.3)))
        if ca > 0:
            for i, (nm, sub, det) in enumerate((("JOHN T. DOWNEY", "AGE 22", "YALE '51  \u00b7  CIA OFFICER"),
                                                ("RICHARD G. FECTEAU", "AGE 25", "BOSTON UNIV.  \u00b7  CIA OFFICER"))):
                k = ease(clamp(remap(t, b["cards"] + i * 0.35, b["cards"] + i * 0.35 + 0.7)), E_OUT)
                x, y = 1340 + (1 - k) * 40, 300 + i * 150
                a = ca * k
                c.drawRect(skia.Rect.MakeXYWH(x - 22, y - 34, 2.0, 108), sk.paint(AMBER, a))
                ui.label(c, nm, x, y, 32, PAPER, a, font_name="cond_m", tracking=0.08)
                ui.label(c, sub, x, y + 36, 17, AMBER, a, tracking=0.25)
                ui.label(c, det, x, y + 64, 17, PAPER, a * 0.7, tracking=0.16)
        # --- site
        sa = clamp(remap(t, b["site"], b["site"] + 0.6))
        if sa > 0:
            s = self.scr(vp, self.site)[0]
            r = 14 + 60 * (1 - ease(clamp(remap(t, b["site"], b["site"] + 1.0)), E_OUT))
            c.drawCircle(s[0], s[1], r, sk.paint(AMBER, sa * 0.9, stroke=1.6))
            c.drawCircle(s[0], s[1], 3.5, sk.paint(AMBER, sa))
            for i in range(3):  # radar-ish pulse
                ph = ((t - b["site"]) * 0.8 + i / 3) % 1
                c.drawCircle(s[0], s[1], 10 + ph * 70, sk.paint(AMBER, sa * 0.35 * (1 - ph), stroke=1.0))
            ui.leader(c, (s[0], s[1]), (s[0] + 120, s[1] - 140), t, b["site"] + 0.2)
            x0, y0 = s[0] + 132, s[1] - 182
            ui.label(c, "PICKUP ZONE", x0, y0, 22, PAPER, sa, tracking=0.25, font_name="mono_m")
            ui.label(c, "SANDAOGOU \u00b7 ANTU COUNTY \u00b7 JILIN", x0, y0 + 30, 18, PAPER, sa * 0.75)
            ui.cjk(c, "吉林省 安图县 三道沟", x0, y0 + 62, 21, PAPER, sa * 0.6)
            ui.label(c, "42\u00b029'07\"N  128\u00b029'40\"E", x0, y0 + 96, 18, AMBER, sa * 0.9)
            ui.label(c, "MOON 95% \u00b7 ALT 60\u00b0", x0, y0 + 122, 17, PAPER, sa * 0.55)
        ui.source_tag(c, t, "N. DUJMOVIC, CIA STUDIES IN INTELLIGENCE 50/4 (2006) \u00b7 DPAA", 1.0, self.dur - 0.2)

    def post(self, t):
        return dict(vignette=0.5, grain=0.04, bloom_str=0.10, halation=0.06)
