"""Line-art 3D: silhouette edges of a mesh, projected wire drawing with depth-cued alpha."""
import numpy as np
from .core import project


class EdgeMesh:
    def __init__(self, V, F):
        self.V = np.asarray(V, np.float64)
        F = np.asarray(F)
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        fid = np.tile(np.arange(len(F)), 3)
        key = np.sort(e, axis=1)
        order = np.lexsort((key[:, 1], key[:, 0]))
        key, fid = key[order], fid[order]
        same = np.all(key[1:] == key[:-1], axis=1)
        pairs_a = np.where(same)[0]
        self.inner = key[pairs_a]
        self.f0, self.f1 = fid[pairs_a], fid[pairs_a + 1]
        mask = np.ones(len(key), bool)
        mask[pairs_a] = False
        mask[pairs_a + 1] = False
        self.boundary = key[mask]
        a, b, c = self.V[F[:, 0]], self.V[F[:, 1]], self.V[F[:, 2]]
        self.fn = np.cross(b - a, c - a)
        self.fc = (a + b + c) / 3

    def silhouette(self, model, cam_world):
        """Edges (N,2,3 world) on the silhouette for the given model matrix and camera position."""
        R, T = model[:3, :3], model[:3, 3]
        inv = np.linalg.inv(model)
        cam_l = inv[:3, :3] @ np.asarray(cam_world) + inv[:3, 3]
        d = cam_l - self.fc
        front = np.einsum("ij,ij->i", self.fn, d) > 0
        sil = self.inner[front[self.f0] != front[self.f1]]
        edges = np.concatenate([sil, self.boundary]) if len(self.boundary) else sil
        p = self.V[edges.reshape(-1)] @ R.T + T
        return p.reshape(-1, 2, 3)


def draw_segments(c, segs_world, vp, paint_fn, w=1920, h=1080):
    """Draw world-space segments (N,2,3). paint_fn(depth)->skia.Paint (bucketed by depth)."""
    import skia
    if len(segs_world) == 0:
        return
    flat = segs_world.reshape(-1, 3)
    s, depth, vis = project(flat, vp, w, h)
    s = s.reshape(-1, 2, 2)
    depth = depth.reshape(-1, 2).mean(1)
    vis = vis.reshape(-1, 2).all(1)
    if not vis.any():
        return
    dmin, dmax = depth[vis].min(), depth[vis].max()
    buckets = 6
    for bkt in range(buckets):
        lo = dmin + (dmax - dmin) * bkt / buckets
        hi = dmin + (dmax - dmin) * (bkt + 1) / buckets + 1e-6
        m = vis & (depth >= lo) & (depth < hi)
        if not m.any():
            continue
        path = skia.Path()
        for (x0, y0), (x1, y1) in s[m]:
            path.moveTo(x0, y0)
            path.lineTo(x1, y1)
        c.drawPath(path, paint_fn(bkt / max(buckets - 1, 1)))


def clip_runs(pts, vp, w=1920, h=1080, near=0.3):
    """Project a 3D polyline, clipping against the camera near plane (w > near). Returns screen runs."""
    p = np.asarray(pts, float)
    ph = np.c_[p, np.ones(len(p))] @ vp.T
    runs, cur = [], []
    def to_s(q):
        return ((q[0] / q[3]) * 0.5 + 0.5) * w, (1 - ((q[1] / q[3]) * 0.5 + 0.5)) * h
    for i in range(len(ph)):
        a = ph[i]
        if a[3] > near:
            if not cur and i > 0 and ph[i - 1][3] <= near:
                b = ph[i - 1]
                k = (near - b[3]) / (a[3] - b[3])
                cur.append(to_s(b + (a - b) * k))
            cur.append(to_s(a))
        else:
            if cur:
                b = ph[i - 1]
                k = (near - b[3]) / (a[3] - b[3])
                cur.append(to_s(b + (a - b) * k))
                runs.append(cur)
                cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) > 1]


def draw_polyline3d(c, pts, vp, paint, w=1920, h=1080):
    import skia
    for r in clip_runs(pts, vp, w, h):
        path = skia.Path()
        path.moveTo(*r[0])
        for q in r[1:]:
            path.lineTo(*q)
        c.drawPath(path, paint)
