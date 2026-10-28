"""Night terrain from real DEM: snow / rock / conifer materials, moon shadows, fog, sky, gunfire lights."""
import math
import numpy as np
import moderngl
from scipy import ndimage
from .geo import DemMosaic

VS_TER = """
#version 330
in vec3 in_pos; in vec3 in_nrm; in float in_forest;
uniform mat4 u_vp; uniform mat4 u_light_vp;
out vec3 v_w; out vec3 v_n; out float v_f; out vec4 v_ls;
void main(){ v_w = in_pos; v_n = in_nrm; v_f = in_forest; v_ls = u_light_vp * vec4(in_pos, 1.0);
  gl_Position = u_vp * vec4(in_pos, 1.0); }
"""

FS_TER = """
#version 330
in vec3 v_w; in vec3 v_n; in float v_f; in vec4 v_ls; out vec4 o;
uniform vec3 u_cam; uniform vec3 u_moon_dir; uniform vec3 u_moon_col; uniform vec3 u_sky_col;
uniform vec3 u_fog_col; uniform float u_fog_density; uniform float u_fog_height;
uniform sampler2D u_shadow; uniform float u_time;
uniform vec3 u_pl_pos[4]; uniform vec3 u_pl_col[4]; uniform sampler2D u_noise; uniform vec2 u_clear_c; uniform float u_clear_r;
float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vn(vec3 x){ return texture(u_noise, x.xz * 0.013 + x.y * 0.0007).r; }
float fbm(vec3 p){ return texture(u_noise, p.xz * 0.011).g; }
float shadow(){
    vec3 p = v_ls.xyz / v_ls.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float s = 0.0; vec2 ts = 1.0 / vec2(textureSize(u_shadow, 0));
    for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++)
        s += (p.z - 0.0008 > texture(u_shadow, p.xy + vec2(x, y) * ts * 1.2).r) ? 0.0 : 1.0;
    return s / 9.0;
}
void main(){
    vec3 n = normalize(v_n);
    float dist = length(u_cam - v_w);
    // micro relief from noise -> perturbed normal (fades with distance)
    float fade = clamp(1.0 - dist / 6000.0, 0.0, 1.0);
    vec2 nq = v_w.xz * 0.0045;
    vec4 nz = texture(u_noise, nq);
    vec3 grad = vec3(nz.b - 0.5, 0.0, nz.a - 0.5);
    n = normalize(n + grad * 1.2 * fade);
    float slope = 1.0 - n.y;
    // conifer forest: clumped canopy with snowy gaps; denser in valleys (in_forest attribute)
    float clump = fbm(v_w * 0.045) * 0.7 + fbm(v_w * 0.22) * 0.5;
    float forest = smoothstep(0.30, 0.80, v_f * 0.8 + clump * 0.55 - 0.22);
    forest *= smoothstep(u_clear_r, u_clear_r * 2.4, length(v_w.xz - u_clear_c) + (texture(u_noise, v_w.xz * 0.008).g - 0.5) * u_clear_r * 1.6);
    float rock = smoothstep(0.55, 0.85, slope + (fbm(v_w * 0.01) - 0.5) * 0.2);
    vec3 snow = vec3(0.86, 0.89, 0.95);
    float tn = texture(u_noise, v_w.xz * 0.09).r * 0.6 + texture(u_noise, v_w.xz * 0.37).r * 0.4;
    vec3 trees = mix(vec3(0.035, 0.045, 0.042), vec3(0.30, 0.32, 0.36), smoothstep(0.55, 0.85, tn));
    vec3 rockc = vec3(0.16, 0.16, 0.17);
    vec3 alb = mix(snow, rockc, rock * 0.7);
    alb = mix(alb, trees, forest * 0.92);
    float sh = shadow();
    // tree self-shadowing makes forest read as volume
    float ndl = max(dot(n, -u_moon_dir), 0.0) * mix(1.0, 0.55 + 0.45 * vn(v_w * 0.7), forest);
    vec3 amb = u_sky_col * (0.55 + 0.45 * n.y);
    vec3 col = alb * (amb + u_moon_col * ndl * sh);
    // snow glints
    vec3 v = normalize(u_cam - v_w);
    vec3 h = normalize(-u_moon_dir + v);
    float g = step(0.9965, hash(floor(v_w * 2.3) + floor(u_time * 0.0))) * pow(max(dot(n, h), 0.0), 40.0);
    col += u_moon_col * g * 6.0 * (1.0 - forest) * sh * fade;
    // gunfire / fire point lights
    for (int i = 0; i < 4; i++){
        vec3 L = u_pl_pos[i] - v_w; float d2 = dot(L, L);
        col += alb * u_pl_col[i] * max(dot(n, normalize(L)), 0.0) / (1.0 + d2 * 0.012);
    }
    float hf = exp(-max(v_w.y - 300.0, 0.0) / u_fog_height);
    float fog = 1.0 - exp(-dist * u_fog_density * (0.25 + 0.75 * hf));
    col = mix(col, u_fog_col, clamp(fog, 0.0, 1.0));
    o = vec4(col, 1.0);
}
"""

VS_TDEPTH = """
#version 330
in vec3 in_pos; uniform mat4 u_light_vp;
void main(){ gl_Position = u_light_vp * vec4(in_pos, 1.0); }
"""

FS_SKY = """
#version 330
in vec2 v_uv; out vec4 o;
uniform mat4 u_inv_vp; uniform vec3 u_moon_dir; uniform float u_time; uniform float u_cloud; uniform sampler2D u_noise;
float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vn(vec3 x){ vec3 i = floor(x), f = fract(x); f = f*f*(3.0-2.0*f);
  return mix(mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x), mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x), mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z); }
float fbm(vec3 p){ return texture(u_noise, p.xz * 0.21).g; }
void main(){
    vec4 a = u_inv_vp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 d = normalize(a.xyz / a.w);
    vec3 m = -u_moon_dir;
    float up = clamp(d.y, -0.2, 1.0);
    vec3 zen = vec3(0.004, 0.008, 0.022), hor = vec3(0.030, 0.045, 0.070);
    vec3 col = mix(hor, zen, pow(max(up, 0.0), 0.45));
    float mu = max(dot(d, m), 0.0);
    col += vec3(0.20, 0.25, 0.34) * pow(mu, 18.0) * 0.6 + vec3(0.05, 0.065, 0.09) * pow(mu, 3.0);
    // stars (suppressed near moon by its glare)
    vec3 sd = d * 420.0;
    float st = step(0.9975, hash(floor(sd)));
    float tw = 0.6 + 0.4 * sin(u_time * 7.0 + hash(floor(sd) + 3.0) * 40.0);
    col += vec3(0.9, 0.95, 1.0) * st * tw * (1.0 - pow(mu, 6.0)) * smoothstep(0.02, 0.2, up) * 0.9;
    // moon disc (0.52 deg) at 95% illumination: soft terminator
    float md = acos(clamp(dot(d, m), -1.0, 1.0));
    float disc = smoothstep(0.0048, 0.0042, md);
    col += vec3(1.0, 0.98, 0.92) * disc * 40.0;
    // thin moonlit cloud
    vec3 cp = d / max(d.y, 0.05) * 1.3 + vec3(u_time * 0.01, 0.0, 0.0);
    float cl = smoothstep(0.52, 0.85, fbm(cp)) * u_cloud * smoothstep(0.0, 0.25, up);
    col = mix(col, vec3(0.10, 0.12, 0.16) + vec3(0.25, 0.3, 0.38) * pow(mu, 8.0), cl * 0.8);
    o = vec4(col, 1.0);
}
"""

VS_STREAK = """
#version 330
in vec3 in_pos; in vec4 in_col; in float in_side;
uniform mat4 u_vp; out vec4 v_col; out float v_side;
void main(){ v_col = in_col; v_side = in_side; gl_Position = u_vp * vec4(in_pos, 1.0); }
"""
FS_STREAK = """
#version 330
in vec4 v_col; in float v_side; out vec4 o;
void main(){ float k = 1.0 - abs(v_side); o = vec4(v_col.rgb * v_col.a * k * k, 0.0); }
"""


def make_noise_tex(gl, size=512, seed=3):
    """Tileable RGBA noise: R value noise, G fbm, B/A gradient-ish detail."""
    r = np.random.default_rng(seed)
    def tile_fbm(octaves, base):
        out = np.zeros((size, size), np.float32)
        amp, tot = 1.0, 0.0
        for o in range(octaves):
            g = base * 2 ** o
            a = r.random((g, g)).astype(np.float32)
            z = ndimage.zoom(a, size / g, order=3, mode="grid-wrap")[:size, :size]
            out += z * amp
            tot += amp
            amp *= 0.5
        out /= tot
        return (out - out.min()) / (out.max() - out.min() + 1e-9)
    R = tile_fbm(1, 64)
    G = tile_fbm(5, 8)
    B = tile_fbm(4, 16)
    A = tile_fbm(4, 16)
    arr = np.dstack([R, G, B, A]).astype("f4")
    tex = gl.ctx.texture((size, size), 4, data=arr.tobytes(), dtype="f4")
    tex.repeat_x = tex.repeat_y = True
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.build_mipmaps()
    return tex


def moon_dir_from(alt_deg, az_deg):
    """Direction the moonlight travels (from moon to ground) in ENU-ish world (x=E, y=up, z=S)."""
    alt, az = math.radians(alt_deg), math.radians(az_deg)
    to_moon = np.array([math.cos(alt) * math.sin(az), math.sin(alt), -math.cos(alt) * math.cos(az)])
    return -to_moon


def _set(p, name, val):
    if name in p:
        if hasattr(val, "tobytes"):
            p[name].write(val.tobytes())
        else:
            p[name].value = val


class TerrainGL:
    def __init__(self, gl, lon, lat, size_m=24000, n=384, z=12, vexag=1.0):
        self.gl = gl
        ctx = gl.ctx
        self.lon, self.lat = lon, lat
        dlat = size_m / 2 / 111320.0
        dlon = size_m / 2 / (111320.0 * math.cos(math.radians(lat)))
        mos = DemMosaic(lon - dlon * 1.05, lat - dlat * 1.05, lon + dlon * 1.05, lat + dlat * 1.05, z)
        xs = np.linspace(-size_m / 2, size_m / 2, n)
        X, Zs = np.meshgrid(xs, xs)  # Zs: south positive
        lons = lon + X / (111320.0 * math.cos(math.radians(lat)))
        lats = lat - Zs / 111320.0
        elev = mos.sample(lons.ravel(), lats.ravel()).reshape(n, n).astype(np.float32)
        elev = ndimage.gaussian_filter(elev, 1.0)
        elev = ndimage.gaussian_filter(elev, 0.6)
        self.elev, self.xs, self.size, self.n = elev, xs, size_m, n
        self.base = float(np.percentile(elev, 5))
        Y = (elev - self.base) * vexag
        self.Y = Y
        cell = size_m / (n - 1)
        gz, gx = np.gradient(Y, cell)
        N = np.dstack([-gx, np.ones_like(Y), -gz])
        N /= np.linalg.norm(N, axis=2, keepdims=True)
        # forest preference: lower slopes / valleys (negative curvature), not on ridgelines
        lap = ndimage.gaussian_filter(Y, 3) - ndimage.gaussian_filter(Y, 12)
        rel = (Y - ndimage.gaussian_filter(Y, 25))
        forest = np.clip(0.65 - lap / 60.0 - rel / 250.0, 0, 1) * np.clip(N[..., 1] * 1.3 - 0.1, 0, 1)
        V = np.dstack([X, Y, Zs]).reshape(-1, 3)
        data = np.c_[V, N.reshape(-1, 3), forest.reshape(-1, 1)].astype("f4")
        idx = np.arange(n * n).reshape(n, n)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
        F = np.stack([np.stack([a, d, c], -1), np.stack([a, c, b], -1)], 2).reshape(-1, 3)
        self.prog = ctx.program(vertex_shader=VS_TER, fragment_shader=FS_TER)
        self.dprog = ctx.program(vertex_shader=VS_TDEPTH, fragment_shader="#version 330\nvoid main(){}")
        vbo = ctx.buffer(data.tobytes())
        ibo = ctx.buffer(F.astype("i4").tobytes())
        self.vao = ctx.vertex_array(self.prog, [(vbo, "3f 3f 1f", "in_pos", "in_nrm", "in_forest")], ibo)
        self.dvao = ctx.vertex_array(self.dprog, [(vbo, "3f 16x", "in_pos")], ibo)
        self.sky = gl.prog("sky", FS_SKY)
        self.noise = make_noise_tex(gl)
        self.streak_prog = ctx.program(vertex_shader=VS_STREAK, fragment_shader=FS_STREAK)

    def height(self, x, zs):
        """Terrain height (scene units) at world x (east), z (south)."""
        n, s = self.n, self.size
        fx = (np.asarray(x) + s / 2) / s * (n - 1)
        fz = (np.asarray(zs) + s / 2) / s * (n - 1)
        return ndimage.map_coordinates(self.Y, [np.atleast_1d(fz), np.atleast_1d(fx)], order=1, mode="nearest")

    def draw_sky(self, fbo, vp, moon_dir, time, cloud=0.5):
        inv = np.linalg.inv(vp)
        self.sky["u_inv_vp"].write(np.asarray(inv, "f4").T.tobytes())
        _set(self.sky, "u_moon_dir", tuple(map(float, moon_dir)))
        _set(self.sky, "u_time", float(time))
        _set(self.sky, "u_cloud", float(cloud))
        self.noise.use(6)
        _set(self.sky, "u_noise", 6)
        fbo.use()
        self.gl.ctx.disable(moderngl.DEPTH_TEST)
        self.gl.ctx.vertex_array(self.sky, []).render(moderngl.TRIANGLES, vertices=3)

    def draw_depth(self, light_vp):
        self.dprog["u_light_vp"].write(np.asarray(light_vp, "f4").T.tobytes())
        self.dvao.render(moderngl.TRIANGLES)

    clear_c = (0.0, 0.0)
    clear_r = 90.0

    def draw(self, vp, cam, light_vp, shadow_tex, moon_dir, time, lights=(), moon_col=(0.55, 0.62, 0.80),
             sky_col=(0.05, 0.065, 0.10), fog_col=(0.045, 0.06, 0.09), fog_density=0.00011, fog_height=900.0):
        p = self.prog
        p["u_vp"].write(np.asarray(vp, "f4").T.tobytes())
        p["u_light_vp"].write(np.asarray(light_vp, "f4").T.tobytes())
        _set(p, "u_cam", tuple(map(float, cam)))
        _set(p, "u_moon_dir", tuple(map(float, moon_dir)))
        _set(p, "u_moon_col", moon_col)
        _set(p, "u_sky_col", sky_col)
        _set(p, "u_fog_col", fog_col)
        _set(p, "u_fog_density", fog_density)
        _set(p, "u_fog_height", fog_height)
        _set(p, "u_time", float(time))
        _set(p, "u_clear_c", tuple(map(float, self.clear_c)))
        _set(p, "u_clear_r", float(self.clear_r))
        pl = list(lights)[:4] + [((0, -1e5, 0), (0, 0, 0))] * (4 - len(list(lights)[:4]))
        p["u_pl_pos"].write(np.array([q[0] for q in pl], "f4").tobytes())
        p["u_pl_col"].write(np.array([q[1] for q in pl], "f4").tobytes())
        shadow_tex.use(5)
        _set(p, "u_shadow", 5)
        self.noise.use(6)
        _set(p, "u_noise", 6)
        self.vao.render(moderngl.TRIANGLES)

    def draw_streaks(self, vp, cam, segs):
        """segs: list of (p0, p1, width, rgb, intensity). Camera-facing additive ribbons."""
        if not segs:
            return
        verts = []
        cam = np.asarray(cam, float)
        for p0, p1, w, rgb, inten in segs:
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            d = p1 - p0
            mid = (p0 + p1) / 2
            side = np.cross(d, cam - mid)
            nn = np.linalg.norm(side)
            if nn < 1e-9:
                continue
            side = side / nn * w
            c = (*rgb, inten)
            quad = [(p0 - side, -1), (p0 + side, 1), (p1 + side, 1), (p0 - side, -1), (p1 + side, 1), (p1 - side, -1)]
            for pos, sd in quad:
                verts.append([*pos, *c, sd])
        data = np.array(verts, "f4")
        ctx = self.gl.ctx
        vbo = ctx.buffer(data.tobytes())
        vao = ctx.vertex_array(self.streak_prog, [(vbo, "3f 4f 1f", "in_pos", "in_col", "in_side")])
        self.streak_prog["u_vp"].write(np.asarray(vp, "f4").T.tobytes())
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.depth_mask = False
        vao.render(moderngl.TRIANGLES)
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        vao.release()
        vbo.release()
