"""Geodata: terrarium DEM tiles, Lambert conformal projection, relief raster, Natural Earth vectors."""
import io
import json
import math
import os
import time
import urllib.request
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = os.path.join(ROOT, "assets", "geo")
DEM = os.path.join(ROOT, "assets", "dem")
CACHE = os.path.join(ROOT, "cache")
os.makedirs(CACHE, exist_ok=True)
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


# ------------------------------------------------------------- DEM tiles
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lr = math.radians(lat)
    y = (1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z, x, y):
    d = os.path.join(DEM, str(z), str(x))
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"{y}.png")
    if not os.path.exists(f):
        for attempt in range(4):
            try:
                with urllib.request.urlopen(TILE_URL.format(z=z, x=x, y=y), timeout=60) as r:
                    data = r.read()
                open(f, "wb").write(data)
                break
            except Exception:
                time.sleep(2 + attempt * 3)
    rgb = np.asarray(Image.open(f).convert("RGB"), np.float32)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


class DemMosaic:
    """Elevation mosaic in Web-Mercator tile space for a lon/lat bbox."""

    def __init__(self, lon0, lat0, lon1, lat1, z):
        self.z = z
        x0, y0 = lonlat_to_tile(lon0, lat1, z)
        x1, y1 = lonlat_to_tile(lon1, lat0, z)
        self.tx0, self.ty0 = int(x0), int(y0)
        self.tx1, self.ty1 = int(x1), int(y1)
        rows = []
        for ty in range(self.ty0, self.ty1 + 1):
            rows.append(np.concatenate([fetch_tile(z, tx, ty) for tx in range(self.tx0, self.tx1 + 1)], 1))
        self.elev = np.concatenate(rows, 0)

    def sample(self, lon, lat):
        lon = np.asarray(lon, np.float64)
        lat = np.clip(np.asarray(lat, np.float64), -85, 85)
        n = 2 ** self.z
        x = (lon + 180.0) / 360.0 * n
        lr = np.radians(lat)
        y = (1.0 - np.log(np.tan(lr) + 1.0 / np.cos(lr)) / np.pi) / 2.0 * n
        px = (x - self.tx0) * 256 - 0.5
        py = (y - self.ty0) * 256 - 0.5
        return ndimage.map_coordinates(self.elev, [py, px], order=1, mode="nearest")


# ------------------------------------------------------------- projection
class MapProj:
    """Lambert conformal conic for NE Asia, mapped to a pixel canvas."""

    def __init__(self, lon0=117.0, lat0=31.0, lon1=143.0, lat1=48.5, width=4096,
                 proj="+proj=lcc +lat_1=35 +lat_2=45 +lat_0=40 +lon_0=128 +ellps=WGS84"):
        self.fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        self.inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
        lons = np.r_[np.linspace(lon0, lon1, 50), np.full(50, lon1), np.linspace(lon1, lon0, 50), np.full(50, lon0)]
        lats = np.r_[np.full(50, lat0), np.linspace(lat0, lat1, 50), np.full(50, lat1), np.linspace(lat1, lat0, 50)]
        X, Y = self.fwd.transform(lons, lats)
        # Use the inscribed rectangle so the canvas never shows outside-bbox corners.
        self.xmin, self.xmax = np.percentile(X, 8), np.percentile(X, 92)
        self.ymin, self.ymax = np.percentile(Y, 8), np.percentile(Y, 92)
        self.width = width
        self.s = width / (self.xmax - self.xmin)
        self.height = int(round((self.ymax - self.ymin) * self.s))
        self.bbox = (lon0, lat0, lon1, lat1)

    def px(self, lon, lat):
        X, Y = self.fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.c_[(np.atleast_1d(X) - self.xmin) * self.s, (self.ymax - np.atleast_1d(Y)) * self.s]

    def lonlat(self, px, py):
        X = np.asarray(px, float) / self.s + self.xmin
        Y = self.ymax - np.asarray(py, float) / self.s
        return self.inv.transform(X, Y)

    def km_per_px(self):
        return 1.0 / self.s / 1000.0


# ------------------------------------------------------------- vectors
def load_lines(name, proj, bbox_pad=2.0, filt=None):
    """Return list of Nx2 pixel arrays for (Multi)LineString/Polygon features inside bbox."""
    lon0, lat0, lon1, lat1 = proj.bbox
    with open(os.path.join(GEO, name + ".geojson")) as fh:
        gj = json.load(fh)
    out = []
    for ft in gj["features"]:
        if filt and not filt(ft["properties"]):
            continue
        g = ft["geometry"]
        if g is None:
            continue
        t = g["type"]
        if t == "LineString":
            parts = [g["coordinates"]]
        elif t == "MultiLineString":
            parts = g["coordinates"]
        elif t == "Polygon":
            parts = g["coordinates"]
        elif t == "MultiPolygon":
            parts = [r for poly in g["coordinates"] for r in poly]
        else:
            continue
        for part in parts:
            a = np.asarray(part, float)[:, :2]
            m = ((a[:, 0] > lon0 - bbox_pad) & (a[:, 0] < lon1 + bbox_pad) &
                 (a[:, 1] > lat0 - bbox_pad) & (a[:, 1] < lat1 + bbox_pad))
            if m.sum() < 2:
                continue
            # split into runs inside the padded bbox
            idx = np.where(m)[0]
            runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for r in runs:
                if len(r) >= 2:
                    out.append((proj.px(a[r, 0], a[r, 1]), ft["properties"]))
    return out


def load_rings(name, proj, min_lat=-60):
    """Whole polygon rings (unclipped) touching the bbox: safe for filled masks."""
    lon0, lat0, lon1, lat1 = proj.bbox
    with open(os.path.join(GEO, name + ".geojson")) as fh:
        gj = json.load(fh)
    out = []
    for ft in gj["features"]:
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                a = np.asarray(ring, float)[:, :2]
                inside = (a[:, 0] > lon0 - 5) & (a[:, 0] < lon1 + 5) & (a[:, 1] > lat0 - 5) & (a[:, 1] < lat1 + 5)
                if not inside.any():
                    continue
                a = a[a[:, 1] > min_lat]
                out.append(proj.px(a[:, 0], a[:, 1]))
    return out


def simplify(pts, tol):
    from shapely.geometry import LineString
    if len(pts) < 3:
        return pts
    return np.asarray(LineString(pts).simplify(tol).coords)


def load_points(proj, names):
    with open(os.path.join(GEO, "ne_10m_populated_places_simple.geojson")) as fh:
        gj = json.load(fh)
    found = {}
    for ft in gj["features"]:
        p = ft["properties"]
        if p.get("name") in names and p.get("name") not in found:
            lon, lat = ft["geometry"]["coordinates"][:2]
            found[p["name"]] = (lon, lat)
    return found


# ------------------------------------------------------------- relief
def hillshade(elev, cell, azimuths=(315, 270, 350, 225), alt=38, zf=1.0):
    gy, gx = np.gradient(elev * zf, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    out = np.zeros_like(elev, dtype=np.float32)
    weights = [0.45, 0.2, 0.2, 0.15]
    for az, w in zip(azimuths, weights):
        a = np.radians(360 - az + 90)
        z = np.radians(90 - alt)
        hs = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
        out += w * np.clip(hs, 0, 1)
    return out


def relief_raster(proj, z=7, key="nea"):
    """Elevation grid sampled on the map canvas (cached .npy)."""
    f = os.path.join(CACHE, f"elev_{key}_{proj.width}.npy")
    if os.path.exists(f):
        return np.load(f)
    lon0, lat0, lon1, lat1 = proj.bbox
    mos = DemMosaic(lon0 - 1, lat0 - 1, lon1 + 1, lat1 + 1, z)
    ys, xs = np.mgrid[0:proj.height, 0:proj.width]
    lon, lat = proj.lonlat(xs.ravel() + 0.5, ys.ravel() + 0.5)
    elev = mos.sample(lon, lat).reshape(proj.height, proj.width).astype(np.float32)
    np.save(f, elev)
    return elev
