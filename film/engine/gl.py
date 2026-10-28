"""Headless OpenGL (EGL / llvmpipe) compositor: HDR layers, blur, bloom, film post."""
import numpy as np
import moderngl
from .core import W, H

VS_FULL = """
#version 330
out vec2 v_uv;
void main(){
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FS_COPY = """
#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_tex; uniform float u_gain;
void main(){ o = texture(u_tex, v_uv) * u_gain; }
"""

# 9-tap separable gaussian using linear filtering (17 effective taps).
FS_BLUR = """
#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_tex; uniform vec2 u_dir; uniform float u_sigma;
void main(){
    vec4 acc = vec4(0.0); float wsum = 0.0;
    for (int i = -8; i <= 8; i++){
        float x = float(i);
        float w = exp(-0.5 * x * x / max(u_sigma * u_sigma, 1e-4));
        acc += texture(u_tex, v_uv + u_dir * x) * w; wsum += w;
    }
    o = acc / wsum;
}
"""

FS_DOWN = """
#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_tex; uniform vec2 u_texel; uniform float u_threshold; uniform int u_first;
vec3 s(vec2 d){ return texture(u_tex, v_uv + d * u_texel).rgb; }
void main(){
    vec3 a = s(vec2(-2,2)), b = s(vec2(0,2)), c = s(vec2(2,2));
    vec3 d = s(vec2(-2,0)), e = s(vec2(0,0)), f = s(vec2(2,0));
    vec3 g = s(vec2(-2,-2)), h = s(vec2(0,-2)), i = s(vec2(2,-2));
    vec3 j = s(vec2(-1,1)), k = s(vec2(1,1)), l = s(vec2(-1,-1)), m = s(vec2(1,-1));
    vec3 col = e*0.125 + (a+c+g+i)*0.03125 + (b+d+f+h)*0.0625 + (j+k+l+m)*0.125;
    if (u_first == 1){
        float br = max(col.r, max(col.g, col.b));
        float knee = u_threshold * 0.6;
        float soft = clamp(br - u_threshold + knee, 0.0, 2.0 * knee);
        soft = soft * soft / (4.0 * knee + 1e-4);
        float contrib = max(soft, br - u_threshold) / max(br, 1e-4);
        col *= contrib;
    }
    o = vec4(col, 1.0);
}
"""

FS_UP = """
#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_tex; uniform sampler2D u_prev; uniform vec2 u_texel; uniform float u_radius;
vec3 s(vec2 d){ return texture(u_tex, v_uv + d * u_texel * u_radius).rgb; }
void main(){
    vec3 col = s(vec2(0))*4.0 + (s(vec2(-1,0))+s(vec2(1,0))+s(vec2(0,-1))+s(vec2(0,1)))*2.0
             + s(vec2(-1,-1))+s(vec2(1,-1))+s(vec2(-1,1))+s(vec2(1,1));
    o = vec4(col / 16.0 + texture(u_prev, v_uv).rgb, 1.0);
}
"""

FS_MIX = """
#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_a; uniform sampler2D u_b; uniform sampler2D u_mask;
uniform float u_t; uniform int u_mode; uniform float u_soft; uniform vec2 u_dir;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }
void main(){
    vec4 a = texture(u_a, v_uv), b = texture(u_b, v_uv);
    float k = u_t;
    if (u_mode == 1){           // luminance-led dissolve: highlights of B arrive first
        float lb = dot(b.rgb, vec3(0.3,0.5,0.2));
        float n = noise(v_uv * vec2(9.0, 5.0)) * 0.35 + lb * 0.65;
        k = smoothstep(n - u_soft, n + u_soft, u_t * (1.0 + 2.0*u_soft) - u_soft);
    } else if (u_mode == 2){    // directional soft wipe with organic edge
        float d = dot(v_uv - 0.5, normalize(u_dir)) + 0.5;
        float n = (noise(v_uv * 18.0) - 0.5) * 0.06;
        k = smoothstep(d + n - u_soft, d + n + u_soft, u_t * (1.0 + 2.0*u_soft) - u_soft);
    } else if (u_mode == 3){    // through black (dip)
        float fa = clamp(1.0 - u_t * 2.0, 0.0, 1.0), fb = clamp(u_t * 2.0 - 1.0, 0.0, 1.0);
        o = a * fa * fa + b * fb * fb; return;
    }
    o = mix(a, b, k);
}
"""

FS_POST = """
#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_hdr; uniform sampler2D u_bloom;
uniform float u_time, u_exposure, u_bloom_str, u_halation, u_grain, u_vignette;
uniform float u_ca, u_distort, u_letterbox, u_fade, u_flash, u_sat, u_contrast;
uniform vec3 u_lift, u_gain, u_tint_shadow, u_tint_high;
uniform vec2 u_res; uniform vec2 u_shake;

float hash(vec3 p){ p = fract(p * 0.1031); p += dot(p, p.yzx + 33.33); return fract((p.x + p.y) * p.z); }

vec3 agx(vec3 x){
    // Compact filmic curve: soft toe, long shoulder, gentle highlight desaturation.
    x = max(x, 0.0);
    vec3 a = x * (2.51 * x + 0.03), b = x * (2.43 * x + 0.59) + 0.14;
    vec3 c = clamp(a / b, 0.0, 1.0);
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float hi = smoothstep(0.75, 1.0, l);
    return mix(c, vec3(l), hi * 0.35);
}

void main(){
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);           // output rows top-down for readback
    uv += u_shake / u_res;
    vec2 c = uv - 0.5; c.x *= u_res.x / u_res.y;
    float r2 = dot(c, c);
    vec2 duv = 0.5 + (uv - 0.5) * (1.0 + u_distort * r2);
    vec2 dir = (duv - 0.5);
    vec3 col;
    col.r = texture(u_hdr, 0.5 + dir * (1.0 + u_ca)).r;
    col.g = texture(u_hdr, duv).g;
    col.b = texture(u_hdr, 0.5 + dir * (1.0 - u_ca)).b;
    vec3 bl = texture(u_bloom, duv).rgb;
    col += bl * u_bloom_str;
    col += bl * vec3(1.0, 0.32, 0.12) * u_halation;
    col *= u_exposure;
    col += vec3(u_flash);
    col = agx(col);
    // split-tone grade
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, col * u_tint_shadow * 1.6, (1.0 - smoothstep(0.0, 0.45, l)) * 0.35);
    col = mix(col, col * u_tint_high * 1.1, smoothstep(0.55, 1.0, l) * 0.25);
    col = (col - 0.5) * u_contrast + 0.5;
    col = mix(vec3(l), col, u_sat);
    col = col * u_gain + u_lift * (1.0 - col);
    // vignette (optical falloff, slightly oval)
    float v = 1.0 - u_vignette * smoothstep(0.25, 1.15, length(c * vec2(0.9, 1.05)));
    col *= v;
    // film grain: two octaves, strongest in mid-tones
    vec2 px = v_uv * u_res;
    float fr = floor(u_time * 24.0);
    float g1 = hash(vec3(floor(px), fr)) - 0.5;
    float g2 = hash(vec3(floor(px * 0.5), fr + 17.0)) - 0.5;
    float g = (g1 * 0.6 + g2 * 0.55);
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col += g * u_grain * (0.35 + 0.65 * (1.0 - abs(lum - 0.45) * 1.6));
    col = clamp(col, 0.0, 1.0) * (1.0 - u_fade);
    // letterbox
    float lb = u_letterbox * 0.5;
    float yy = v_uv.y;
    float bar = smoothstep(lb, lb + 0.0015, yy) * (1.0 - smoothstep(1.0 - lb - 0.0015, 1.0 - lb, yy));
    col *= bar;
    // triangular dither before 8-bit quantisation
    float d = hash(vec3(px, fr + 3.0)) + hash(vec3(px + 7.0, fr)) - 1.0;
    col += d / 255.0;
    o = vec4(col, 1.0);
}
"""

VS_PLANE = """
#version 330
in vec3 in_pos; in vec2 in_uv;
uniform mat4 u_mvp;
out vec2 v_uv;
void main(){ v_uv = in_uv; gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""

FS_PLANE = """
#version 330
in vec2 v_uv; out vec4 o;
uniform sampler2D u_tex; uniform float u_opacity; uniform vec3 u_tint; uniform float u_gain;
void main(){
    vec4 c = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y));
    o = vec4(c.rgb * u_tint * u_gain, c.a) * u_opacity;
}
"""

POST_DEFAULT = dict(
    exposure=1.0, bloom_str=0.08, halation=0.05, grain=0.045, vignette=0.42,
    ca=0.0018, distort=0.025, letterbox=0.0, fade=0.0, flash=0.0, sat=0.92, contrast=1.04,
    lift=(0.010, 0.012, 0.016), gain=(1.0, 0.985, 0.95),
    tint_shadow=(0.82, 0.94, 1.08), tint_high=(1.04, 1.0, 0.92), shake=(0.0, 0.0),
    bloom_threshold=0.85,
)


class GL:
    def __init__(self, w=W, h=H):
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.gc_mode = "auto"
        self.w, self.h = w, h
        self.vao_empty = self.ctx.vertex_array(self.prog("copy"), [])
        self._pool = {}
        self.plane_prog = self.ctx.program(vertex_shader=VS_PLANE, fragment_shader=FS_PLANE)
        q = np.array([-.5, -.5, 0, 0, 0, .5, -.5, 0, 1, 0, .5, .5, 0, 1, 1,
                      -.5, -.5, 0, 0, 0, .5, .5, 0, 1, 1, -.5, .5, 0, 0, 1], "f4")
        self.plane_vbo = self.ctx.buffer(q.tobytes())
        self.plane_vao = self.ctx.vertex_array(self.plane_prog, [(self.plane_vbo, "3f 2f", "in_pos", "in_uv")])

    # ----------------------------------------------------------- resources
    _progs = {}

    def prog(self, name, fs=None):
        key = (id(self.ctx), name)
        if key not in GL._progs:
            src = fs or {"copy": FS_COPY, "blur": FS_BLUR, "down": FS_DOWN, "up": FS_UP,
                         "mix": FS_MIX, "post": FS_POST}[name]
            GL._progs[key] = self.ctx.program(vertex_shader=VS_FULL, fragment_shader=src)
        return GL._progs[key]

    def target(self, key, w=None, h=None, dtype="f2", comps=4, samples=0, depth=False):
        w, h = w or self.w, h or self.h
        k = (key, w, h, dtype, samples, depth)
        if k not in self._pool:
            tex = self.ctx.texture((w, h), comps, dtype=dtype, samples=samples)
            if not samples:
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                tex.repeat_x = tex.repeat_y = False
            att = [tex]
            dep = self.ctx.depth_renderbuffer((w, h), samples=samples) if depth else None
            fbo = self.ctx.framebuffer(color_attachments=att, depth_attachment=dep)
            self._pool[k] = (fbo, tex)
        return self._pool[k]

    def texture_rgba8(self, arr, mipmap=False, tex=None):
        """Upload an HxWx4 uint8 premultiplied RGBA array (top-left origin)."""
        h, w = arr.shape[:2]
        if tex is None or tex.size != (w, h):
            tex = self.ctx.texture((w, h), 4, data=np.ascontiguousarray(arr).tobytes())
        else:
            tex.write(np.ascontiguousarray(arr).tobytes())
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR) if mipmap else (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        if mipmap:
            tex.build_mipmaps()
            tex.anisotropy = 8.0
        return tex

    def texture_float(self, arr, comps=None):
        h, w = arr.shape[:2]
        comps = comps or (arr.shape[2] if arr.ndim == 3 else 1)
        tex = self.ctx.texture((w, h), comps, data=np.ascontiguousarray(arr, "f4").tobytes(), dtype="f4")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    # ----------------------------------------------------------- passes
    def fullscreen(self, prog, fbo, textures=(), **uniforms):
        fbo.use()
        for i, (name, tex) in enumerate(textures):
            tex.use(i)
            if name in prog:
                prog[name].value = i
        for k, v in uniforms.items():
            if k in prog:
                prog[k].value = v
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        self.ctx.vertex_array(prog, []).render(moderngl.TRIANGLES, vertices=3)

    def blur(self, tex, sigma, key="blur", scale_down=1):
        """Separable gaussian; sigma in source pixels. Downscales for large sigma."""
        w, h = tex.size
        ds = scale_down
        while sigma / ds > 6.0:
            ds *= 2
        bw, bh = max(1, w // ds), max(1, h // ds)
        fa, ta = self.target(key + "a", bw, bh)
        fb, tb = self.target(key + "b", bw, bh)
        p = self.prog("blur")
        s = max(sigma / ds, 0.3)
        step = max(1.0, s / 3.0)
        self.fullscreen(self.prog("copy"), fa, [("u_tex", tex)], u_gain=1.0)
        self.fullscreen(p, fb, [("u_tex", ta)], u_dir=(step / bw, 0.0), u_sigma=s / step)
        self.fullscreen(p, fa, [("u_tex", tb)], u_dir=(0.0, step / bh), u_sigma=s / step)
        return ta

    def bloom(self, hdr_tex, threshold=0.85, levels=6):
        w, h = self.w // 2, self.h // 2
        downs = []
        src = hdr_tex
        sw, sh = self.w, self.h
        for i in range(levels):
            f, t = self.target(f"bd{i}", w, h)
            self.fullscreen(self.prog("down"), f, [("u_tex", src)], u_texel=(1.0 / sw, 1.0 / sh),
                            u_threshold=threshold, u_first=1 if i == 0 else 0)
            downs.append((f, t, w, h))
            src, sw, sh = t, w, h
            w, h = max(1, w // 2), max(1, h // 2)
        prev = downs[-1][1]
        for i in range(levels - 2, -1, -1):
            f, t, w, h = downs[i]
            fu, tu = self.target(f"bu{i}", w, h)
            self.fullscreen(self.prog("up"), fu, [("u_tex", prev), ("u_prev", t)],
                            u_texel=(1.0 / w, 1.0 / h), u_radius=1.0)
            prev = tu
        return prev

    def mix(self, a, b, t, mode=0, soft=0.15, direction=(1.0, 0.0), key="mix"):
        f, tex = self.target(key)
        self.fullscreen(self.prog("mix"), f, [("u_a", a), ("u_b", b)], u_t=float(t), u_mode=mode,
                        u_soft=soft, u_dir=direction)
        return tex

    def post(self, hdr_tex, time, **kw):
        p = dict(POST_DEFAULT)
        p.update(kw)
        bloom = self.bloom(hdr_tex, threshold=p.pop("bloom_threshold"))
        f, tex = self.target("final", dtype="f1")
        uni = {"u_" + k: v for k, v in p.items()}
        self.fullscreen(self.prog("post"), f, [("u_hdr", hdr_tex), ("u_bloom", bloom)],
                        u_time=float(time), u_res=(float(self.w), float(self.h)), **uni)
        return f

    def read_rgb(self, fbo):
        return fbo.read(components=3, alignment=1)

    # ----------------------------------------------------------- 3D planes
    def draw_plane(self, tex, mvp, opacity=1.0, tint=(1, 1, 1), gain=1.0, additive=False):
        self.ctx.enable(moderngl.BLEND)
        if additive:
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        else:
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(0)
        self.plane_prog["u_tex"].value = 0
        self.plane_prog["u_mvp"].write(np.asarray(mvp, "f4").T.tobytes())
        self.plane_prog["u_opacity"].value = float(opacity)
        self.plane_prog["u_tint"].value = tuple(tint)
        self.plane_prog["u_gain"].value = float(gain)
        self.plane_vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)
