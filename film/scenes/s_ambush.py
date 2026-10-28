"""S4 - Ambush reconstruction at the DPAA crash site, Antu County, Jilin. Night of 29 Nov 1952.

Geography: terrain is centred on the DPAA excavation datum (the crash site). The pickup zone sits
~120 m up-track; the plane is hit on its pass over the zone, pulls up, then glides into the trees at the datum.
"""
import math
import numpy as np
import moderngl
from engine import c47, sk, ui
from engine.core import (W, H, PAPER, clamp, remap, ease, E_SOFT, E_INOUT, lerp, look_at, perspective, project,
                         translate, rot_x, rot_z, hash1)
from engine.terrain import TerrainGL, moon_dir_from
from engine.mesh3d import MeshGL, ShadowMap, ortho
from scenes.base import Scene

SITE = (128.4945, 42.4852)
MOON_ALT, MOON_AZ = 61.0, 227.0  # ephem, ~23:00 local 29 Nov 1952
PZ = -120.0                      # pickup zone x (m) relative to the crash datum
V = 32.0                         # ~62 knots
FOG = dict(fog_col=(0.04, 0.055, 0.085), fog_density=0.00013, fog_height=700.0)


def sn(t, seed):
    return math.sin(t * 1.7 + seed) * 0.5 + math.sin(t * 3.1 + seed * 2.3) * 0.3 + math.sin(t * 7.3 + seed * 0.7) * 0.2


class AmbushScene(Scene):
    name = "ambush"

    def setup(self):
        gl = self.gl
        self.ter = TerrainGL(gl, SITE[0], SITE[1], size_m=12000, n=440, z=12)
        self.ter.clear_c = (PZ, 0.0)
        self.ter.clear_r = 70.0
        self.gz = lambda x, z=0.0: float(self.ter.height(x, z)[0])
        self.h0 = self.gz(PZ)
        Vv, N, F = c47.build_c47()
        self.plane = MeshGL(gl, Vv, N, F)
        q = np.array([[-1.1, 0, -1.6], [1.1, 0, -1.6], [1.1, 0, 1.6], [-1.1, 0, 1.6]], "f4")
        self.sheet = MeshGL(gl, q, np.tile([0, 1, 0], (4, 1)).astype("f4"), np.array([[0, 1, 2], [0, 2, 3]]))
        self.shadow = ShadowMap(gl, 2048)
        self.moon = moon_dir_from(MOON_ALT, MOON_AZ)
        self.b = dict(guns=5.2, hit=5.7, crash=8.4)
        self.b.update(self.p.get("beats", {}))
        self.guns = [np.array([PZ + 25.0, 0, 48.0]), np.array([PZ - 15.0, 0, -45.0])]
        for g in self.guns:
            g[1] = self.gz(g[0], g[2]) + 0.6
        self.fires = [np.array([PZ + dx, self.gz(PZ + dx, dz) + 0.6, dz]) for dx, dz in [(-38, 9), (2, -11), (40, 9)]]
        self.people = [np.array([PZ - 7 + i * 2.4, 0, 12 + (i % 2) * 2.5]) for i in range(5)]
        for p in self.people:
            p[1] = self.gz(p[0], p[2])
        # descent tuned so impact lands on the crash beat at the datum
        kc = self.b["crash"] - self.b["hit"]
        self.x_hit = PZ  # hit over the pickup zone
        xc = self.x_hit + V * kc
        self.vx = (0.0 - self.x_hit) / kc if xc > 40 else V
        self.crash = np.array([self.x_hit + self.vx * kc, 0, self.z_off(kc)])
        self.crash[1] = self.gz(self.crash[0], self.crash[2]) + 3
        self.ov = sk.Canvas(W, H)
        self.ovtex = None
        lv = look_at(-self.moon * 4000, np.zeros(3), (0, 1, 0))
        self.light_vp = ortho(-2500, 2500, -2500, 2500, 100, 9000) @ lv

    def z_off(self, k):
        return 10 * clamp(k / 2.7) ** 2

    def plane_state(self, t):
        b = self.b
        k = t - b["hit"]
        if k < 0:
            x = self.x_hit + V * k
            ground = max(self.gz(x), self.h0)
            alt = ground + 16.0 + max(0.0, -k - 1.5) * 3.0
            return np.array([x, alt, 0.0]), translate(x, alt, 0.0) @ rot_z(0.03), 0.03
        kc = b["crash"] - b["hit"]
        k = min(k, kc)
        x = self.x_hit + self.vx * k
        z = self.z_off(k)
        ground = self.gz(x, z)
        climb = 20.0 * (1 - math.exp(-1.8 * k))
        u = clamp((k - 0.8) / max(kc - 0.8, 0.1))
        start = self.h0 + 16.0 + 20.0 * (1 - math.exp(-1.8 * kc))
        drop = (start - (ground + 3.0)) * u ** 1.7
        alt = self.h0 + 16.0 + climb - drop
        pitch = 0.18 * math.exp(-1.6 * k) - 0.10 * u
        roll = -0.18 * clamp(k - 0.5, 0, 2.0)
        pos = np.array([x, alt, z])
        return pos, translate(*pos) @ rot_z(pitch) @ rot_x(roll), pitch

    def shot(self, t):
        b = self.b
        pos, _, _ = self.plane_state(min(t, b["crash"]))
        zone = np.array([PZ, self.h0, 0.0])
        if t < 2.7:      # A: establishing - the lit zone in a snowy valley, plane approaching, exhausts glowing
            k = t / 2.7
            eye = zone + np.array([-420 + 70 * k, 120 - 15 * k, 330])
            tgt = lerp(zone + np.array([-60, 12, 0]), pos, 0.35)
            fov = 30
        elif t < b["guns"]:   # B: tracking alongside, low and slow
            k = (t - 2.7) / (b["guns"] - 2.7)
            eye = pos + np.array([-30 + 12 * k, 6.0, 58 - 6 * k])
            tgt = pos + np.array([4, -4, 0])
            fov = 33
        elif t < b["hit"] + 1.3:   # C: low by gun 1, looking up across the crossfire
            g = self.guns[0]
            eye = g + np.array([-34, 2.4, 38])
            tgt = lerp(pos, g, 0.25) + np.array([0, 4, 0])
            fov = 40
        else:                       # D: from the slope, plane pulls up and glides into the trees; hold on the glow
            k = clamp((t - b["hit"] - 1.3) / 5.0)
            eye = np.array([PZ - 140 + 40 * k, self.h0 + 48, 330 - 25 * k])
            tgt = lerp(pos + np.array([10, 0, 0]), self.crash + np.array([0, 8, 0]), ease(clamp((t - b["crash"] + 0.6) / 1.5), E_SOFT))
            fov = 32
        if b["guns"] < t < b["hit"] + 1.3:
            eye = eye + np.array([sn(t * 4, 1), sn(t * 4, 2), 0]) * 0.25
        return eye, tgt, fov

    def tracers(self, t):
        b = self.b
        segs, lights = [], []
        if t < b["guns"] or t > b["crash"] + 1.2:
            return segs, lights
        for gi, g in enumerate(self.guns):
            muzzle = g + np.array([0, 1.2, 0])
            rate = 12.0
            n_now = int((t - b["guns"]) * rate)
            for n in range(max(0, n_now - 10), n_now + 1):
                t_fire = b["guns"] + n / rate + gi * 0.04
                age = t - t_fire
                if age < 0 or age > 0.8:
                    continue
                aim, _, _ = self.plane_state(min(t_fire + 0.2, b["crash"]))
                spread = np.array([hash1(n * 7 + gi) - 0.5, hash1(n * 11 + gi) - 0.4, hash1(n * 13 + gi) - 0.5]) * 22
                d = aim + spread - muzzle
                d /= np.linalg.norm(d)
                head = muzzle + d * 800 * age
                tail = muzzle + d * max(0, 800 * age - 22)
                segs.append((tail, head, 0.11, (1.0, 0.40, 0.12), 13.0 * (1 - age / 0.8)))
            fl = max(0.0, math.sin(t * 75 + gi * 2.1))
            lights.append((muzzle + [0, 2, 0], (60 * fl, 26 * fl, 8 * fl)))
            segs.append((muzzle, muzzle + np.array([0.6, 1.1, 0.3]) * fl, 0.7, (1.0, 0.75, 0.4), 30 * fl))
        return segs, lights

    def draw(self, t, fbo):
        gl = self.gl
        ctx = gl.ctx
        b = self.b
        eye, tgt, fov = self.shot(t)
        vp = perspective(fov, W / H, 2.0, 30000) @ look_at(eye, tgt)
        pos, m, pitch = self.plane_state(min(t, b["crash"]))
        show_plane = t < b["crash"]
        self.shadow.begin()
        ctx.enable(moderngl.DEPTH_TEST)
        self.ter.draw_depth(self.light_vp)
        if show_plane:
            self.plane.draw_depth(self.light_vp, m)
        ctx.disable(moderngl.DEPTH_TEST)
        fm, tm = gl.target("amb_ms", samples=4, depth=True)
        fm.use()
        fm.clear(0, 0, 0, 1, depth=1.0)
        self.ter.draw_sky(fm, vp, self.moon, t, cloud=0.18)
        fm.use()
        ctx.enable(moderngl.DEPTH_TEST)
        segs, lights = self.tracers(t)
        fires = []
        for i, f in enumerate(self.fires):
            fl = 0.75 + 0.25 * sn(t * 6, i * 3)
            fires.append((f + [0, 1.2, 0], (3.2 * fl, 1.4 * fl, 0.45 * fl)))
        if t > b["crash"] - 0.05:
            k = t - b["crash"]
            fl = (0.85 + 0.15 * sn(t * 9, 5)) * (160 * math.exp(-k * 2.5) + 30)
            fires.insert(0, (self.crash + [0, 6, 0], (fl, fl * 0.42, fl * 0.12)))
        lights_all = (lights + fires)[:4]
        self.ter.draw(vp, eye, self.light_vp, self.shadow.tex, self.moon, t, lights=lights_all,
                      moon_col=(0.40, 0.48, 0.68), sky_col=(0.05, 0.065, 0.10), **FOG)
        if show_plane:
            gg = sum(l[1][0] for l in lights) * 0.0015
            self.plane.draw(vp, eye, model=m, ldir=self.moon, lcol=(0.55, 0.62, 0.80),
                            amb_top=(0.035, 0.045, 0.07), amb_bot=(0.02 + gg, 0.016 + gg * 0.45, 0.012 + gg * 0.1),
                            albedo=(0.085, 0.095, 0.07), spec=0.55, rough=0.32, rim=0.22, **FOG)
        for gi, g in enumerate(self.guns):
            k = max(0.0, t - b["guns"] - gi * 0.07)
            if k < 2.2:
                sm = translate(*(g + np.array([2.5 * k, 0.3 + 7 * k + 4 * k * k, -9 * k]))) @ rot_x(k * 4.0 + gi) @ rot_z(k * 2.5)
                self.sheet.draw(vp, eye, model=sm, ldir=self.moon, lcol=(0.55, 0.62, 0.80), amb_top=(0.07, 0.08, 0.11),
                                amb_bot=(0.03, 0.03, 0.04), albedo=(0.82, 0.83, 0.85), spec=0.05, rough=0.9, **FOG)
        if show_plane:
            R, T = m[:3, :3], m[:3, 3]
            for ex in c47.EXHAUST:
                p0 = R @ np.array(ex) + T
                segs.append((p0, p0 - R[:, 0] * 1.4, 0.16, (1.0, 0.5, 0.2), 3.0 + 1.5 * sn(t * 20, ex[2])))
            if t > b["hit"]:
                eng = R @ np.array(c47.EXHAUST[1]) + T
                k = t - b["hit"]
                for i in range(12):
                    a = hash1(i * 3.1 + int(t * 24))
                    off = np.array([-(3 + a * 16), (hash1(i * 5.7 + int(t * 24)) - 0.3) * 3, (hash1(i * 9.1) - 0.5) * 3])
                    segs.append((eng + off * 0.5, eng + off, 0.25, (1.0, 0.45, 0.12), 9 * clamp(k * 2)))
            if b["hit"] - 0.1 < t < b["hit"] + 1.0:
                for i in range(18):
                    if hash1(i + int(t * 24) * 13) > 0.5:
                        p0 = T + np.array([(hash1(i * 2.3) - 0.5) * 14, (hash1(i * 4.1) - 0.3) * 3, (hash1(i * 6.7) - 0.5) * 8])
                        dv = np.array([hash1(i * 8.1) - 0.5, hash1(i * 3.3), hash1(i * 5.5) - 0.5]) * 2.5
                        segs.append((p0, p0 + dv, 0.12, (1.0, 0.85, 0.55), 26))
        for i, f in enumerate(self.fires):
            fl = 0.75 + 0.25 * sn(t * 6, i * 3)
            segs.append((f, f + np.array([0, 1.3 * fl, 0]), 0.55, (1.0, 0.55, 0.2), 7 * fl))
        if t > b["crash"] - 0.05:
            k = t - b["crash"]
            c0 = self.crash
            segs.append((c0 - [0, 3, 0], c0 + np.array([0, 9 + 5 * sn(t * 5, 1), 0]), 6.0, (1.0, 0.45, 0.12),
                         8 + 40 * math.exp(-k * 2.5)))
            for i in range(18):
                ph = (t * 0.4 + hash1(i * 1.7)) % 1.0
                p0 = c0 + np.array([(hash1(i * 4.4) - 0.5) * 16, ph * 45, (hash1(i * 7.2) - 0.5) * 16])
                segs.append((p0, p0 + np.array([0.3, 1.0, 0]), 0.14, (1.0, 0.6, 0.2), 10 * (1 - ph)))
        self.ter.draw_streaks(vp, eye, segs)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.copy_framebuffer(fbo, fm)
        c = self.ov.clear().c
        self.overlay(c, t, vp)
        self.ovtex = gl.texture_rgba8(self.ov.flush(), tex=self.ovtex)
        fbo.use()
        gl.draw_plane(self.ovtex, np.diag([2, 2, 1, 1]))

    def overlay(self, c, t, vp):
        b = self.b
        dark = (0.012, 0.013, 0.018)
        for z in (7.6, -7.6):
            gy = self.gz(PZ, z)
            s0 = project(np.array([[PZ, gy, z]]), vp)[0][0]
            s1 = project(np.array([[PZ, gy + 4.6, z]]), vp)[0][0]
            hgt = abs(s1[1] - s0[1])
            if 1 < hgt < 300:
                c.drawLine(*s0, *s1, sk.paint(dark, 0.9, stroke=max(1.0, hgt * 0.035)))
        for p in self.people:
            s0 = project(p[None], vp)[0][0]
            s1 = project((p + [0, 1.75, 0])[None], vp)[0][0]
            hgt = s0[1] - s1[1]
            if 1.0 < hgt < 90:
                w = max(1.0, hgt * 0.2)
                c.drawRoundRect(skia_rect(s1[0] - w / 2, s1[1] + hgt * 0.16, w, hgt * 0.84), w * 0.3, w * 0.3, sk.paint(dark, 0.95))
                c.drawCircle(s1[0], s1[1] + hgt * 0.08, max(0.8, hgt * 0.08), sk.paint(dark, 0.95))
        a = ui.fade_in_out(t, 0.5, 4.6, 0.6, 0.6)
        if a > 0:
            ui.label(c, "RECONSTRUCTION", 72, 156, 15, PAPER, 0.55 * a, tracking=0.4)
            ui.label(c, "SANDAOGOU, ANTU COUNTY, JILIN  \u00b7  NIGHT OF 29 NOV 1952", 72, 184, 18, PAPER, 0.85 * a, tracking=0.16)
            ui.label(c, "MOON 95% ILLUMINATED  \u00b7  ALT 61\u00b0  \u00b7  SPEED ~60 KNOTS", 72, 210, 15, PAPER, 0.5 * a, tracking=0.16)
        ui.caption(c, t, "CHINESE ACCOUNT: THE MAN IN THE HARNESS WAS A SECURITY OFFICER POSING AS THE COURIER",
                   72, H - 150, 2.9, b["guns"] - 0.2, size=14)
        ui.caption(c, t, "SECOND PASS  \u00b7  TWO GUNS UNDER WHITE SHEETS", 72, H - 150, b["guns"] + 0.1, b["crash"] - 0.2, size=14)
        ui.caption(c, t, "TERRAIN: REAL ELEVATION DATA AT THE DPAA CRASH-SITE COORDINATES  42\u00b029'07\"N 128\u00b029'40\"E",
                   72, H - 150, b["crash"] + 0.4, self.dur - 0.3, size=14)

    def post(self, t):
        b = self.b
        flash = 0.12 * math.exp(-(t - b["crash"]) * 6) if b["crash"] - 0.02 < t < b["crash"] + 0.8 else 0.0
        fade = clamp(remap(t, self.dur - 0.8, self.dur))
        shake = (sn(t * 9, 3) * 3.5, sn(t * 8, 4) * 2.0) if b["guns"] < t < b["hit"] + 1.0 else (0.0, 0.0)
        return dict(exposure=1.45, bloom_str=0.20, halation=0.12, bloom_threshold=0.75, vignette=0.55, grain=0.06,
                    letterbox=0.115, flash=flash, fade=fade, tint_shadow=(0.78, 0.92, 1.12), sat=0.85, shake=shake)


def skia_rect(x, y, w, h):
    import skia
    return skia.Rect.MakeXYWH(x, y, w, h)
