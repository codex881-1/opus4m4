"""S3 - Blueprint: how the All American air pickup worked, then the trap."""
import math
import numpy as np
import skia
import moderngl
from engine import sk, ui, c47
from engine.core import (W, H, PAPER, AMBER, RED, clamp, remap, ease, E_OUT, E_INOUT, E_SOFT, lerp, Track,
                         look_at, perspective, project, rot_y, translate)
from engine.lineart import EdgeMesh, draw_segments, draw_polyline3d
from engine.mesh3d import MeshGL
from scenes.base import Scene

BG = (0.018, 0.026, 0.040)
POLE_H = 4.57          # 15 ft
POLE_Z = 7.62          # 50 ft apart
CABLE = 15.24          # 50 ft
ALT = 17.4             # ~57 ft (keeps grapple at line height)
LINE_Y = 4.0
MAN = np.array([-8.0, 0.0, 0.0])
DOOR_L = np.array([c47.S0 - 14.6, -0.55, -1.2])   # left aft cargo door (local)
GUNS = [np.array([-4.0, 0.0, 24.0]), np.array([7.0, 0.0, -23.0])]  # schematic, straddling path


class BlueprintScene(Scene):
    name = "blueprint"

    def setup(self):
        V, N, F = c47.build_c47()
        self.edges = EdgeMesh(V, F)
        self.mesh = MeshGL(self.gl, V, N, F)
        self.lines = c47.blueprint_lines()
        self.l0 = sk.Canvas(W, H)
        self.l1 = sk.Canvas(W, H)
        self.t0tex = self.t1tex = None
        b = dict(poles=0.0, plane=2.0, catch=3.7, winch=4.6, rewind=6.9, trap=7.3, guns=8.4)
        b.update(self.p.get("beats", {}))
        self.b = b
        self.v = (70 + 15.64) / (b["catch"] - b["plane"] + 0.7)

    # ----------------------------------------------------------- kinematics
    def plane_x(self, t):
        b = self.b
        x = 70 - self.v * (t - (b["plane"] - 0.7))
        if t > b["rewind"]:  # rewind to the approach for the trap reveal
            k = ease(remap(t, b["rewind"], b["trap"] + 0.3), E_INOUT)
            xe = 70 - self.v * (b["rewind"] - (b["plane"] - 0.7))
            x = lerp(xe, 34.0, k)
        return x

    def model(self, t):
        return translate(self.plane_x(t), ALT, 0) @ rot_y(math.pi)

    def door(self, t):
        m = self.model(t)
        return m[:3, :3] @ DOOR_L + m[:3, 3]

    def grapple(self, t):
        d = self.door(t)
        ang = math.radians(36)
        return d + np.array([math.sin(ang) * CABLE, -math.cos(ang) * CABLE, 1.2])

    def lift(self, t):
        b = self.b
        if t > b["rewind"]:
            return 0.0
        return ease(remap(t, b["catch"], b["catch"] + 0.55), E_OUT)

    def man_pos(self, t):
        b = self.b
        k = self.lift(t)
        if k <= 0:
            return MAN.copy()
        d = self.door(t)
        L = lerp(CABLE + 7.0, 1.0, ease(remap(t, b["winch"], b["winch"] + 2.1), E_INOUT))
        ang = math.radians(lerp(20, 58, k))
        trail = d + np.array([math.sin(ang) * L, -math.cos(ang) * L, 0.6])
        return lerp(MAN + [0, 0.0, 0], trail, k)

    def camera(self, t):
        b = self.b
        px = self.plane_x(t)
        follow = ease(remap(t, b["catch"] - 0.1, b["catch"] + 0.9), E_INOUT) * (1 - ease(remap(t, b["rewind"], b["trap"]), E_INOUT))
        tx = lerp(-1.0, px + 13.0, follow)
        trap = ease(remap(t, b["trap"], b["guns"] + 0.4), E_INOUT)
        push = ease(clamp(t / 4.0), E_SOFT)
        eye = np.array([tx + lerp(18, 10, push), lerp(5.0, 5.5, push), lerp(60, 52, push)])
        tgt = np.array([tx - 2.0, 9.0, 0.0])
        eye_t = np.array([4.0, 74.0, 62.0])
        tgt_t = np.array([4.0, 0.0, -2.0])
        eye = lerp(eye, eye_t, trap)
        tgt = lerp(tgt, tgt_t, trap)
        return perspective(lerp(33, 34, trap), W / H, 0.5, 2000) @ look_at(eye, tgt), eye

    # ----------------------------------------------------------- draw
    def draw(self, t, fbo):
        gl = self.gl
        b = self.b
        trap = clamp(remap(t, b["trap"], b["trap"] + 0.6))
        bg = tuple(lerp(x, y, trap * 0.5) for x, y in zip(BG, (0.03, 0.012, 0.012)))
        fbo.use()
        fbo.clear(*bg, 1.0)
        vp, eye = self.camera(t)
        line = tuple(lerp(x, y, trap * 0.0) for x, y in zip(PAPER, PAPER))
        # layer 0: ground grid + rig + guns
        c = self.l0.clear().c
        self.draw_grid(c, vp, eye, t, trap)
        self.t0tex = gl.texture_rgba8(self.l0.flush(), tex=self.t0tex)
        gl.draw_plane(self.t0tex, np.diag([2, 2, 1, 1]))
        # plane solid fill (occludes grid), drawn with depth into a fresh depth buffer
        pa = clamp(remap(t, b["plane"] - 0.7, b["plane"] - 0.3))
        if pa > 0:
            fd, td = gl.target("bp_depth", depth=True)
            fd.use()
            fd.clear(0, 0, 0, 0, depth=1.0)
            gl.ctx.enable(moderngl.DEPTH_TEST)
            self.mesh.draw(vp, eye, model=self.model(t), albedo=(0.0, 0.0, 0.0), amb_top=(0.0, 0.0, 0.0),
                           amb_bot=(0, 0, 0), lcol=(0, 0, 0), spec=0.0, rim=0.0,
                           emit=tuple(x * 0.9 for x in bg), alpha=pa)
            gl.ctx.disable(moderngl.DEPTH_TEST)
            fbo.use()
            gl.ctx.enable(moderngl.BLEND)
            gl.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            gl.draw_plane(td, np.diag([2, -2, 1, 1]))
        # layer 1: plane lines + cable + man + annotations
        c = self.l1.clear().c
        self.draw_plane_lines(c, vp, eye, t, pa, trap)
        self.draw_rig_dynamic(c, vp, t, trap)
        self.annotate(c, vp, t, trap)
        self.t1tex = gl.texture_rgba8(self.l1.flush(), tex=self.t1tex)
        gl.draw_plane(self.t1tex, np.diag([2, 2, 1, 1]))

    def S(self, vp, p):
        s, d, v = project(np.atleast_2d(p), vp)
        return s

    def draw_grid(self, c, vp, eye, t, trap):
        a = 0.10 * clamp(remap(t, 0, 0.8))
        for i in range(-30, 31):
            x = i * 5.0
            al = a * (1.6 if i % 4 == 0 else 1.0)
            draw_polyline3d(c, [(x, 0, -60), (x, 0, 60)], vp, sk.paint(PAPER, al, stroke=1.0))
        for j in range(-12, 13):
            z = j * 5.0
            al = a * (1.6 if j % 4 == 0 else 1.0)
            draw_polyline3d(c, [(-150, 0, z), (150, 0, z)], vp, sk.paint(PAPER, al, stroke=1.0))
        # flight path centreline
        fp = clamp(remap(t, self.b["plane"] - 0.5, self.b["plane"] + 0.5))
        col = tuple(lerp(x, y, trap) for x, y in zip(AMBER, RED))
        draw_polyline3d(c, [(120, 0.02, 0), (-120, 0.02, 0)], vp,
                        sk.dashed(sk.paint(col, 0.45 * fp, stroke=1.4), 10, 8, -t * 40))
        # guns + crossfire fans (trap)
        ga = clamp(remap(t, self.b["guns"], self.b["guns"] + 0.5))
        if ga > 0:
            for gi, g in enumerate(GUNS):
                s = self.S(vp, g)[0]
                r = 10 + 40 * (1 - ease(clamp(remap(t, self.b["guns"] + gi * 0.15, self.b["guns"] + gi * 0.15 + 0.6)), E_OUT))
                c.drawCircle(s[0], s[1], r, sk.paint(RED, ga, stroke=2.0))
                c.drawLine(s[0] - r - 8, s[1], s[0] + r + 8, s[1], sk.paint(RED, ga, stroke=1.2))
                c.drawLine(s[0], s[1] - r - 8, s[0], s[1] + r + 8, sk.paint(RED, ga, stroke=1.2))
                fk = ease(clamp(remap(t, self.b["guns"] + 0.5, self.b["guns"] + 1.3)), E_OUT)
                for k in range(9):
                    tgt = np.array([lerp(30, -30, k / 8), ALT * 0.8, 0])
                    p1 = g + (tgt - g) * fk
                    draw_polyline3d(c, [g + [0, 0.5, 0], p1], vp, sk.paint(RED, ga * 0.35, stroke=1.0))

    def draw_rig_dynamic(self, c, vp, t, trap):
        b = self.b
        a = clamp(remap(t, b["poles"] + 0.1, b["poles"] + 0.5))
        if a <= 0:
            return
        grow = ease(remap(t, b["poles"] + 0.1, b["poles"] + 0.9), E_OUT)
        col = PAPER
        pp = sk.paint(col, a, stroke=2.4)
        for z in (POLE_Z, -POLE_Z):
            draw_polyline3d(c, [(0, 0, z), (0, POLE_H * grow, z)], vp, pp)
            s = self.S(vp, (0, 0, z))[0]
            c.drawCircle(s[0], s[1], 4, sk.paint(col, a, stroke=1.2))
        lk = ease(remap(t, b["poles"] + 0.7, b["poles"] + 1.5), E_OUT)
        caught = t >= b["catch"] and t < b["rewind"]
        man = self.man_pos(t)
        if lk > 0:
            if not caught:
                zs = np.linspace(-POLE_Z, POLE_Z, 24)
                sag = LINE_Y + (POLE_H - LINE_Y) * (zs / POLE_Z) ** 2
                pts = np.c_[np.zeros_like(zs), sag, zs]
                n = max(2, int(len(pts) * lk))
                draw_polyline3d(c, pts[:n], vp, sk.paint(col, a, stroke=1.6))
                if lk > 0.95:
                    draw_polyline3d(c, [(0, LINE_Y, 0), man + [0, 1.25, 0]], vp, sk.paint(col, a * 0.9, stroke=1.3))
            else:
                g = self.grapple(t) if self.lift(t) < 1 else None
                hook = self.door(t)
                draw_polyline3d(c, [hook, man + [0, 1.25, 0]], vp, sk.paint(AMBER, 0.95, stroke=1.8))
        # trajectory trace of the lift
        if b["catch"] < t < b["rewind"]:
            ts = np.linspace(b["catch"], min(t, b["winch"] + 2.1), 40)
            tr = np.array([self.man_pos(x) for x in ts])
            draw_polyline3d(c, tr + [0, 1.0, 0], vp, sk.dashed(sk.paint(AMBER, 0.5, stroke=1.2), 6, 6))
        # man (billboard silhouette)
        self.draw_man(c, vp, man, a, trap)
        # contact flash
        if b["catch"] - 0.02 < t < b["catch"] + 0.3:
            s = self.S(vp, (0, LINE_Y, 0))[0]
            k = remap(t, b["catch"], b["catch"] + 0.3)
            c.drawCircle(s[0], s[1], 6 + 40 * k, sk.paint(AMBER, (1 - k) * 0.9, stroke=2.0))

    def draw_man(self, c, vp, p, a, trap):
        top = self.S(vp, p + [0, 1.78, 0])[0]
        bot = self.S(vp, p)[0]
        hpx = max(8.0, bot[1] - top[1])
        x, y = bot
        col = tuple(lerp(u, v, trap) for u, v in zip(PAPER, RED))
        s = hpx / 1.78
        pth = skia.Path()
        pth.addCircle(x, y - 1.62 * s, 0.13 * s)
        body = skia.Path()
        body.moveTo(x - 0.22 * s, y - 1.42 * s)
        body.lineTo(x + 0.22 * s, y - 1.42 * s)
        body.lineTo(x + 0.26 * s, y - 0.80 * s)
        body.lineTo(x + 0.16 * s, y - 0.80 * s)
        body.lineTo(x + 0.14 * s, y)
        body.lineTo(x + 0.02 * s, y)
        body.lineTo(x, y - 0.62 * s)
        body.lineTo(x - 0.02 * s, y)
        body.lineTo(x - 0.14 * s, y)
        body.lineTo(x - 0.16 * s, y - 0.80 * s)
        body.lineTo(x - 0.26 * s, y - 0.80 * s)
        body.close()
        c.drawPath(pth, sk.paint(col, a * 0.95))
        c.drawPath(body, sk.paint(col, a * 0.95))
        # harness straps
        hp = sk.paint(AMBER if trap < 0.5 else RED, a, stroke=max(1.0, 0.035 * s))
        c.drawLine(x - 0.2 * s, y - 1.38 * s, x + 0.18 * s, y - 0.86 * s, hp)
        c.drawLine(x + 0.2 * s, y - 1.38 * s, x - 0.18 * s, y - 0.86 * s, hp)

    def draw_plane_lines(self, c, vp, eye, t, pa, trap):
        if pa <= 0:
            return
        m = self.model(t)
        sil = self.edges.silhouette(m, eye)
        col = PAPER
        draw_segments(c, sil, vp, lambda k: sk.paint(col, pa * (0.95 - 0.35 * k), stroke=1.6))
        R, T = m[:3, :3], m[:3, 3]
        for ln in self.lines:
            draw_polyline3d(c, np.asarray(ln) @ R.T + T, vp, sk.paint(col, pa * 0.28, stroke=1.0))
        # props: discs
        for pc in c47.PROP_CENTERS:
            ctr = R @ np.array(pc) + T
            ang = np.linspace(0, 2 * np.pi, 48)
            ring = np.c_[np.zeros(48), np.cos(ang) * c47.PROP_RADIUS, np.sin(ang) * c47.PROP_RADIUS]
            ring = ring @ R.T + ctr
            draw_polyline3d(c, ring, vp, sk.dashed(sk.paint(col, pa * 0.4, stroke=1.0), 4, 4, t * 200))
        # trailing cable + grapple (before catch / after rewind)
        b = self.b
        if t < b["catch"] or t > b["rewind"]:
            dk = ease(remap(t, b["plane"] - 0.2, b["plane"] + 0.8), E_OUT) if t < b["rewind"] else 1.0
            d = self.door(t)
            g = self.grapple(t)
            gp = d + (g - d) * dk
            draw_polyline3d(c, [d, gp], vp, sk.paint(AMBER, pa, stroke=1.6))
            s = self.S(vp, gp)[0]
            for k in range(4):
                a = math.radians(-60 + k * 40)
                c.drawLine(s[0], s[1], s[0] + math.cos(a) * 9, s[1] + math.sin(a) * 9 - 4, sk.paint(AMBER, pa, stroke=1.4))

    def annotate(self, c, vp, t, trap):
        b = self.b
        na = 1 - clamp(remap(t, b["rewind"] - 0.3, b["rewind"]))
        # poles: dimension lines
        top = [self.S(vp, (0, POLE_H, z))[0] for z in (POLE_Z, -POLE_Z)]
        base = [self.S(vp, (0, 0, z))[0] for z in (POLE_Z, -POLE_Z)]
        if na > 0:
            ui.dim_line(c, base[0], top[0], "15 FT", t, b["poles"] + 0.9, a=0.85 * na, off=-18, size=16)
            ui.dim_line(c, base[0], base[1], "~50 FT", t, b["poles"] + 1.2, a=0.85 * na, off=22, size=16)
            mpt = self.S(vp, MAN + [0, 1.9, 0])[0]
            la = clamp(remap(t, b["poles"] + 1.3, b["poles"] + 1.7)) * (1 - clamp(remap(t, b["catch"], b["catch"] + 0.3)))
            if la > 0:
                ui.leader(c, (mpt[0], mpt[1]), (mpt[0] - 60, mpt[1] - 70), t, b["poles"] + 1.3, a=0.8 * la)
                ui.label(c, "AGENT IN HARNESS", mpt[0] - 64, mpt[1] - 80, 18, PAPER, la, align="right", tracking=0.18)
        # plane data
        pa = clamp(remap(t, b["plane"], b["plane"] + 0.5)) * na
        if pa > 0:
            nose = self.S(vp, self.model(t)[:3, 3] + [0, 4.5, 0])[0]
            ui.label(c, "C-47  \u00b7  ~60 KNOTS  \u00b7  ~50 FT", nose[0] - 40, nose[1] - 30, 18, PAPER, pa, tracking=0.16)
            g = self.S(vp, self.grapple(t))[0]
            ga = clamp(remap(t, b["plane"] + 0.6, b["plane"] + 1.0)) * (1 - clamp(remap(t, b["catch"], b["catch"] + 0.2)))
            if ga > 0:
                ui.label(c, "50-FT STEEL CABLE + GRAPPLE", g[0] + 20, g[1] + 30, 17, AMBER, ga, tracking=0.14)
        wa = clamp(remap(t, b["winch"], b["winch"] + 0.4)) * na
        if wa > 0:
            d = self.S(vp, self.door(t))[0]
            ui.leader(c, (d[0], d[1]), (d[0] + 40, d[1] - 90), t, b["winch"], a=0.8 * wa)
            ui.label(c, "WINCH AT THE REAR DOOR", d[0] + 46, d[1] - 118, 18, AMBER, wa, tracking=0.16)
            ui.label(c, "OPERATORS: DOWNEY, FECTEAU", d[0] + 46, d[1] - 94, 17, PAPER, wa * 0.8, tracking=0.14)
        # title block
        tb = clamp(remap(t, 0.3, 1.0)) * (1 - clamp(remap(t, self.dur - 0.4, self.dur)))
        if tb > 0:
            x, y = W - 520, H - 150
            c.drawRect(skia.Rect.MakeXYWH(x, y, 456, 92), sk.paint(PAPER, 0.5 * tb, stroke=1.0))
            c.drawLine(x, y + 46, x + 456, y + 46, sk.paint(PAPER, 0.3 * tb, stroke=1.0))
            ui.label(c, "ALL AMERICAN AIR PICKUP SYSTEM", x + 16, y + 30, 17, PAPER, 0.9 * tb, tracking=0.14)
            ui.label(c, "WWII DESIGN \u00b7 SCHEMATIC, NOT TO SCALE", x + 16, y + 74, 14, PAPER, 0.6 * tb, tracking=0.14)
        sc = clamp(remap(t, b["plane"], b["plane"] + 0.4)) * (1 - clamp(remap(t, b["rewind"] - 0.3, b["rewind"])))
        if sc > 0:
            ui.label(c, "TIME \u00d7 1.0", 64, 80, 14, PAPER, 0.45 * sc, tracking=0.3)
        # rewind marker
        if b["rewind"] < t < b["trap"] + 0.4:
            k = remap(t, b["rewind"], b["trap"] + 0.4)
            ui.label(c, "REWIND", W / 2, 120, 22, PAPER, math.sin(k * math.pi) * 0.8, align="center", tracking=0.6)
        # trap annotations
        ta = clamp(remap(t, b["trap"], b["trap"] + 0.4))
        if ta > 0:
            mpt = self.S(vp, MAN + [0, 2.2, 0])[0]
            ui.leader(c, (mpt[0], mpt[1]), (mpt[0] - 80, mpt[1] - 110), t, b["trap"], rgb=RED, a=0.9 * ta)
            ui.label(c, "TEAM ALREADY CAPTURED", mpt[0] - 86, mpt[1] - 146, 20, RED, ta, align="right",
                     font_name="mono_sb", tracking=0.16)
            ui.label(c, "RADIO IN CHINESE HANDS", mpt[0] - 86, mpt[1] - 120, 18, PAPER, ta * 0.85, align="right", tracking=0.16)
            ui.cjk(c, "电台已被中方控制", mpt[0] - 86, mpt[1] - 92, 18, RED, ta * 0.7, align="right")
        ga = clamp(remap(t, b["guns"] + 0.3, b["guns"] + 0.8))
        if ga > 0:
            for gi, g in enumerate(GUNS):
                s = self.S(vp, g)[0]
                ui.label(c, "AA GUN %d" % (gi + 1), s[0] + 34, s[1] - 10, 18, RED, ga, tracking=0.2, font_name="mono_sb")
                ui.label(c, "UNDER WHITE SHEETS", s[0] + 34, s[1] + 14, 15, PAPER, ga * 0.75, tracking=0.16)
            ui.label(c, "CROSSFIRE ACROSS THE FLIGHT PATH", W / 2, 110, 22, RED, ga, align="center", tracking=0.24)
        ui.source_tag(c, t, "DUJMOVIC (CIA, 2006) \u00b7 W. LEARY, CIA'S AIR PICKUP (1994) \u00b7 CIA.GOV", 0.5, self.dur - 0.2)

    def post(self, t):
        return dict(vignette=0.48, grain=0.035, bloom_str=0.06, halation=0.03, ca=0.0012)
