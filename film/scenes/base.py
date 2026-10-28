"""Timeline: scenes placed on the master clock with transitions."""
import moderngl
from engine.core import FPS, INK


class Scene:
    """Base scene. Subclasses implement setup() and draw(t, fbo); t is local seconds."""
    name = "scene"

    def __init__(self, gl, dur, **params):
        self.gl = gl
        self.dur = dur
        self.p = params
        self.setup()

    def setup(self):
        pass

    def draw(self, t, fbo):
        fbo.use()
        fbo.clear(*INK, 1.0)

    def post(self, t):
        return {}


class Timeline:
    def __init__(self, gl):
        self.gl = gl
        self.items = []  # (start, scene, transition_in)

    def add(self, start, scene, trans=None):
        """trans: dict(dur=seconds, mode=0 cross|1 luma|2 wipe|3 dip, soft=.., dir=(x,y)) overlapping previous."""
        self.items.append((start, scene, trans))
        return self

    def total_frames(self):
        return int(round(max(s + sc.dur for s, sc, _ in self.items) * FPS))

    def _render_scene(self, idx, T):
        start, sc, _ = self.items[idx]
        fbo, tex = self.gl.target(f"scene{idx % 2}")
        sc.draw(T - start, fbo)
        self.gl.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        return tex, sc.post(T - start)

    def render(self, T):
        from render import blend_post
        active = [i for i, (s, sc, _) in enumerate(self.items) if s <= T < s + sc.dur]
        if not active:
            fbo, tex = self.gl.target("scene0")
            fbo.use()
            fbo.clear(0, 0, 0, 1)
            return tex, {"fade": 1.0}
        if len(active) == 1:
            return self._render_scene(active[0], T)
        a, b = active[-2], active[-1]
        sb, scb, tr = self.items[b]
        tr = tr or {"dur": 0.0, "mode": 0}
        k = (T - sb) / max(tr["dur"], 1e-6)
        ta, pa = self._render_scene(a, T)
        # copy A out of the shared slot before B renders
        fa, keep = self.gl.target("keepA")
        self.gl.fullscreen(self.gl.prog("copy"), fa, [("u_tex", ta)], u_gain=1.0)
        tb, pb = self._render_scene(b, T)
        mixed = self.gl.mix(keep, tb, min(max(k, 0.0), 1.0), mode=tr.get("mode", 0), soft=tr.get("soft", 0.15),
                            direction=tr.get("dir", (1.0, 0.0)))
        return mixed, blend_post(pa, pb, min(max(k, 0.0), 1.0))
