"""Timing, easing and small math helpers shared by every scene."""
import math
import numpy as np

W, H = 1920, 1080
FPS = 24

# Palette (linear-ish floats, tuned for the grade in post).
INK = (0.016, 0.020, 0.027)
PAPER = (0.905, 0.872, 0.795)
PAPER_DIM = (0.62, 0.60, 0.55)
AMBER = (1.0, 0.66, 0.26)
RED = (0.86, 0.13, 0.09)
STEEL = (0.62, 0.70, 0.76)
MOON = (0.70, 0.80, 1.0)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def remap(x, a, b):
    """0..1 progress of x through [a, b], clamped."""
    if b == a:
        return 1.0 if x >= b else 0.0
    return clamp((x - a) / (b - a))


def smoothstep(a, b, x):
    t = remap(x, a, b)
    return t * t * (3 - 2 * t)


def mix3(a, b, t):
    return tuple(lerp(x, y, t) for x, y in zip(a, b))


class Bezier:
    """CSS-style cubic-bezier easing (x1, y1, x2, y2)."""

    def __init__(self, x1, y1, x2, y2):
        self.p = (x1, y1, x2, y2)

    def _b(self, t, a, b):
        return 3 * a * t * (1 - t) ** 2 + 3 * b * t * t * (1 - t) + t ** 3

    def _db(self, t, a, b):
        return 3 * a * (1 - t) ** 2 + 6 * (b - a) * t * (1 - t) + 3 * (1 - b) * t * t

    def __call__(self, x):
        x = clamp(x)
        x1, y1, x2, y2 = self.p
        t = x
        for _ in range(8):
            d = self._b(t, x1, x2) - x
            dd = self._db(t, x1, x2)
            if abs(d) < 1e-6 or dd == 0:
                break
            t = clamp(t - d / dd)
        return self._b(t, y1, y2)


# A small, deliberate easing vocabulary used everywhere for consistency.
E_OUT = Bezier(0.16, 1.0, 0.3, 1.0)        # expo-like settle
E_INOUT = Bezier(0.65, 0.0, 0.35, 1.0)     # camera moves
E_IN = Bezier(0.7, 0.0, 0.84, 0.0)         # accelerate away
E_SOFT = Bezier(0.45, 0.05, 0.25, 1.0)     # gentle reveal
E_LINEARISH = Bezier(0.3, 0.1, 0.7, 0.9)


def ease(x, curve=E_OUT):
    return curve(clamp(x))


def spring(t, freq=2.2, damp=5.5):
    """Damped spring step response, 0 -> 1 with a little overshoot."""
    if t <= 0:
        return 0.0
    return 1 - math.exp(-damp * t) * math.cos(2 * math.pi * freq * t)


class Track:
    """Keyframed scalar/vector track: [(time, value, curve_to_next), ...]."""

    def __init__(self, keys):
        self.keys = sorted(keys, key=lambda k: k[0])

    def __call__(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        for i in range(len(ks) - 1):
            t0, v0 = ks[i][0], ks[i][1]
            t1, v1 = ks[i + 1][0], ks[i + 1][1]
            if t <= t1:
                curve = ks[i][2] if len(ks[i]) > 2 else E_INOUT
                u = curve((t - t0) / (t1 - t0))
                if isinstance(v0, (tuple, list, np.ndarray)):
                    return tuple(lerp(a, b, u) for a, b in zip(v0, v1))
                return lerp(v0, v1, u)
        return ks[-1][1]


def hash1(n):
    """Deterministic pseudo-random in [0,1) from an int/float seed."""
    x = math.sin(n * 12.9898 + 78.233) * 43758.5453
    return x - math.floor(x)


def rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- 3D camera
def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, np.float64), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def project(points, view_proj, w=W, h=H):
    """Project Nx3 world points -> (Nx2 screen px, N depth, N visible mask)."""
    p = np.asarray(points, np.float64)
    ph = np.c_[p, np.ones(len(p))] @ view_proj.T
    wv = ph[:, 3:4]
    ndc = ph[:, :3] / np.where(np.abs(wv) < 1e-9, 1e-9, wv)
    sx = (ndc[:, 0] * 0.5 + 0.5) * w
    sy = (1 - (ndc[:, 1] * 0.5 + 0.5)) * h
    return np.c_[sx, sy], ph[:, 3], (wv[:, 0] > 1e-6)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scale(sx, sy=None, sz=None):
    sy = sx if sy is None else sy
    sz = sx if sz is None else sz
    return np.diag([sx, sy, sz, 1.0])
