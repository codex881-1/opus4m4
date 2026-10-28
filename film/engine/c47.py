"""Parametric Douglas C-47 (DC-3) mesh from published dimensions.

Frame: +X forward (nose), +Y up, +Z right wing. Origin near the wing quarter-chord.
Length 19.43 m, span 28.96 m, NACA 2215 root -> 2206 tip, LE sweep ~15 deg, straight TE,
~5 deg dihedral on the outer panels, R-1830 nacelles, 3.53 m props.
"""
import math
import numpy as np

S0 = 7.6  # station offset: x = S0 - s (s measured aft from the nose)

# station s, half-width, half-height, centre height
_FUS = [
    (0.00, 0.02, 0.02, -0.15), (0.12, 0.30, 0.34, -0.12), (0.40, 0.55, 0.62, -0.08),
    (0.90, 0.82, 0.92, -0.04), (1.60, 1.02, 1.16, 0.00), (2.40, 1.15, 1.32, 0.03),
    (3.40, 1.23, 1.41, 0.04), (5.00, 1.26, 1.44, 0.04), (9.50, 1.26, 1.44, 0.04),
    (11.5, 1.19, 1.35, 0.12), (13.3, 1.02, 1.14, 0.30), (15.0, 0.80, 0.88, 0.50),
    (16.6, 0.58, 0.62, 0.68), (18.0, 0.36, 0.38, 0.82), (19.0, 0.16, 0.18, 0.92),
    (19.43, 0.03, 0.04, 0.96),
]


def _interp_fus(s):
    a = np.array(_FUS)
    return (np.interp(s, a[:, 0], a[:, 1]), np.interp(s, a[:, 0], a[:, 2]), np.interp(s, a[:, 0], a[:, 3]))


def naca4(m, p, t, n=28):
    """Closed NACA 4-digit airfoil loop (upper TE->LE->lower TE), unit chord, x aft."""
    beta = np.linspace(0, math.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    loop_x = np.r_[xu[::-1], xl[1:]]
    loop_y = np.r_[yu[::-1], yl[1:]]
    return loop_x, loop_y


class Mesh:
    def __init__(self):
        self.v = []
        self.f = []
        self.parts = {}

    def add_grid(self, grid, name, wrap_u=False, cap_ends=False):
        """grid: (Nv, Nu, 3). Quads between neighbours; wrap_u closes the loop direction."""
        nv, nu, _ = grid.shape
        base = sum(len(x) for x in self.v)
        self.v.append(grid.reshape(-1, 3))
        faces = []
        uu = nu if wrap_u else nu - 1
        for i in range(nv - 1):
            for j in range(uu):
                a = base + i * nu + j
                b = base + i * nu + (j + 1) % nu
                c = base + (i + 1) * nu + (j + 1) % nu
                d = base + (i + 1) * nu + j
                faces += [(a, b, c), (a, c, d)]
        if cap_ends:
            for row, flip in ((0, True), (nv - 1, False)):
                cidx = base + nv * nu + (0 if row == 0 else 1)
                faces_cap = []
                for j in range(uu):
                    a = base + row * nu + j
                    b = base + row * nu + (j + 1) % nu
                    faces_cap.append((cidx, b, a) if flip else (cidx, a, b))
                faces += faces_cap
            self.v.append(np.array([grid[0].mean(0), grid[-1].mean(0)]))
        start = sum(len(x) for x in self.f)
        self.f.append(np.array(faces, np.int32))
        self.parts[name] = (start, start + len(faces))

    def build(self):
        V = np.concatenate(self.v).astype(np.float32)
        F = np.concatenate(self.f).astype(np.int32)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        N = np.zeros_like(V)
        for k in range(3):
            np.add.at(N, F[:, k], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        return V, N, F


def fuselage_grid(ns=70, nu=56):
    s = (1 - np.cos(np.linspace(0, math.pi, ns))) / 2 * 19.43
    u = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    g = np.zeros((ns, nu, 3))
    for i, si in enumerate(s):
        w, h, yc = _interp_fus(si)
        c, sn = np.cos(u), np.sin(u)
        e = 2.0 / 2.4  # superellipse: slightly boxy, DC-3 sections are near-round
        y = np.sign(sn) * np.abs(sn) ** e * h
        y = np.where(y < 0, y * 0.94, y)  # flatter belly
        z = np.sign(c) * np.abs(c) ** e * w
        g[i, :, 0] = S0 - si
        g[i, :, 1] = yc + y
        g[i, :, 2] = z
    return g


def wing_grid(side=1, nspan=26):
    """Low wing: centre section flat to |z|=3.9, outer panels 5 deg dihedral."""
    zs = np.r_[np.linspace(0.0, 3.9, 6), np.linspace(3.9, 14.30, nspan)[1:], 14.42, 14.48]
    le_root, chord_root, chord_tip = 6.05, 4.30, 1.55
    te_s = le_root + chord_root  # straight trailing edge
    rows = []
    for z in zs:
        if z <= 3.9:
            chord = chord_root
            dy = 0.0
            t = 0.15
        else:
            k = (z - 3.9) / (14.48 - 3.9)
            chord = chord_root + (chord_tip - chord_root) * k
            dy = (z - 3.9) * math.tan(math.radians(5.0))
            t = 0.15 + (0.06 - 0.15) * k
        if z > 14.30:  # rounded tip
            r = (z - 14.30) / (14.48 - 14.30)
            chord *= math.sqrt(max(0.0, 1 - r * r)) * 0.9 + 0.1
            t *= 0.6
        lx, ly = naca4(0.02, 0.2, t)
        le = te_s - chord
        x = S0 - (le + lx * chord)
        y = -0.95 + dy + ly * chord
        rows.append(np.c_[x, y, np.full_like(x, z * side)])
    g = np.array(rows)
    if side < 0:
        g = g[:, ::-1]
    return g


def tail_h_grid(side=1):
    zs = np.linspace(0.0, 4.06, 14)
    rows = []
    for z in zs:
        k = z / 4.06
        chord = 2.75 + (1.35 - 2.75) * k
        if k > 0.9:
            chord *= math.sqrt(max(0.0, 1 - ((k - 0.9) / 0.1) ** 2)) * 0.85 + 0.15
        lx, ly = naca4(0.0, 0.3, 0.10)
        te = 19.25
        le = te - chord - 0.35 * k
        x = S0 - (le + lx * chord)
        y = 0.62 + ly * chord
        rows.append(np.c_[x, y, np.full_like(x, z * side)])
    g = np.array(rows)
    if side < 0:
        g = g[:, ::-1]
    return g


def tail_v_grid():
    hs = np.linspace(0.0, 2.75, 16)
    rows = []
    for hgt in hs:
        k = hgt / 2.75
        chord = 4.2 * (1 - k) ** 1.6 + 1.25 * k  # long curved dorsal fillet -> rounded top
        if k > 0.88:
            chord *= math.sqrt(max(0.0, 1 - ((k - 0.88) / 0.12) ** 2)) * 0.8 + 0.2
        te = 19.35 - 0.15 * k
        le = te - chord
        lx, ly = naca4(0.0, 0.3, 0.09)
        x = S0 - (le + lx * chord)
        rows.append(np.c_[x, np.full_like(x, 1.05 + hgt), ly * chord])
    return np.array(rows)


def nacelle_grid(side=1, ns=30, nu=36):
    zc, yc = 3.6 * side, -0.62
    s = np.linspace(0, 1, ns)
    st = 3.25 + s * (10.2 - 3.25)
    prof = np.interp(st, [3.25, 3.32, 3.6, 4.1, 5.1, 6.3, 8.0, 9.4, 10.2],
                     [0.30, 0.62, 0.71, 0.72, 0.68, 0.60, 0.45, 0.22, 0.03])
    drop = np.interp(st, [3.25, 6.0, 8.5, 10.2], [0.0, -0.05, -0.18, -0.25])
    u = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    g = np.zeros((ns, nu, 3))
    for i in range(ns):
        g[i, :, 0] = S0 - st[i]
        g[i, :, 1] = yc + drop[i] + np.sin(u) * prof[i] * np.where(np.sin(u) < 0, 1.15, 1.0)
        g[i, :, 2] = zc + np.cos(u) * prof[i]
    return g


def build_c47():
    m = Mesh()
    m.add_grid(fuselage_grid(), "fuselage", wrap_u=True)
    for sd in (1, -1):
        m.add_grid(wing_grid(sd), f"wing{sd}", wrap_u=True)
        m.add_grid(tail_h_grid(sd), f"htail{sd}", wrap_u=True)
        m.add_grid(nacelle_grid(sd), f"nacelle{sd}", wrap_u=True)
    m.add_grid(tail_v_grid(), "vtail", wrap_u=True)
    return m.build()


PROP_CENTERS = [(S0 - 3.20, -0.62, 3.6), (S0 - 3.20, -0.62, -3.6)]
PROP_RADIUS = 1.765
EXHAUST = [(S0 - 5.4, -1.25, 3.6 + 0.35), (S0 - 5.4, -1.25, -3.6 - 0.35)]
HOOK_MOUNT = (S0 - 12.0, -1.35, 0.0)  # belly, aft cabin (pickup arm attach, illustrative)


def blueprint_lines():
    """Feature polylines (world coords) for line-art: section rings, wing outline, windows."""
    lines = []
    g = fuselage_grid(ns=70, nu=56)
    for idx in (6, 14, 24, 36, 48, 58, 64):
        lines.append(np.vstack([g[idx], g[idx][:1]]))
    for sd in (1, -1):
        w = wing_grid(sd)
        lines.append(w[:, 0])                     # trailing edge
        le_idx = w.shape[1] // 2
        lines.append(w[:, le_idx])                # leading edge
        lines.append(np.vstack([w[-1], w[-1][:1]]))
        n = nacelle_grid(sd)
        for idx in (1, 3, 8, 16):
            lines.append(np.vstack([n[idx], n[idx][:1]]))
        h = tail_h_grid(sd)
        lines.append(h[:, 0])
        lines.append(h[:, h.shape[1] // 2])
        # cabin windows: 7 per side
        for k in range(7):
            s = 5.6 + k * 1.02
            w_, h_, yc = _interp_fus(s)
            x0, x1 = S0 - s, S0 - s - 0.42
            y0, y1 = yc + 0.15, yc + 0.62
            zz = (w_ + 0.005) * sd
            lines.append(np.array([[x0, y0, zz], [x1, y0, zz], [x1, y1, zz], [x0, y1, zz], [x0, y0, zz]]))
    v = tail_v_grid()
    lines.append(v[:, 0])
    lines.append(v[:, v.shape[1] // 2])
    return lines
