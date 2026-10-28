"""Lit mesh rendering (moonlight / studio), shadow maps, fog. GLSL 330 on llvmpipe."""
import numpy as np
import moderngl

VS_MESH = """
#version 330
in vec3 in_pos; in vec3 in_nrm;
uniform mat4 u_vp; uniform mat4 u_model; uniform mat4 u_light_vp;
out vec3 v_w; out vec3 v_n; out vec4 v_ls;
void main(){
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_w = w.xyz; v_n = mat3(u_model) * in_nrm;
    v_ls = u_light_vp * w;
    gl_Position = u_vp * w;
}
"""

FS_MESH = """
#version 330
in vec3 v_w; in vec3 v_n; in vec4 v_ls; out vec4 o;
uniform vec3 u_cam; uniform vec3 u_ldir; uniform vec3 u_lcol; uniform vec3 u_amb_top; uniform vec3 u_amb_bot;
uniform vec3 u_albedo; uniform float u_spec; uniform float u_rough; uniform float u_rim;
uniform vec3 u_fog_col; uniform float u_fog_density; uniform float u_fog_height;
uniform sampler2D u_shadow; uniform int u_use_shadow; uniform vec3 u_emit; uniform float u_alpha;
float shadow(){
    if (u_use_shadow == 0) return 1.0;
    vec3 p = v_ls.xyz / v_ls.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 1.0;
    float s = 0.0; vec2 ts = 1.0 / vec2(textureSize(u_shadow, 0));
    for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++){
        float d = texture(u_shadow, p.xy + vec2(x, y) * ts * 1.5).r;
        s += (p.z - 0.0015 > d) ? 0.0 : 1.0;
    }
    return s / 9.0;
}
void main(){
    vec3 n = normalize(v_n);
    vec3 v = normalize(u_cam - v_w);
    if (dot(n, v) < 0.0) n = -n;
    vec3 l = normalize(-u_ldir);
    float ndl = max(dot(n, l), 0.0);
    float sh = shadow();
    vec3 h = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), mix(8.0, 180.0, 1.0 - u_rough)) * u_spec;
    float fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
    vec3 amb = mix(u_amb_bot, u_amb_top, n.y * 0.5 + 0.5);
    vec3 col = u_albedo * (amb + u_lcol * ndl * sh) + u_lcol * spec * sh + fres * u_rim * u_amb_top * 2.0 + u_emit;
    float dist = length(u_cam - v_w);
    float hf = exp(-max(v_w.y, 0.0) / max(u_fog_height, 1.0));
    float fog = 1.0 - exp(-dist * u_fog_density * (0.35 + 0.65 * hf));
    col = mix(col, u_fog_col, clamp(fog, 0.0, 1.0));
    o = vec4(col * u_alpha, u_alpha);
}
"""

VS_DEPTH = """
#version 330
in vec3 in_pos; uniform mat4 u_light_vp; uniform mat4 u_model;
void main(){ gl_Position = u_light_vp * u_model * vec4(in_pos, 1.0); }
"""
FS_DEPTH = """
#version 330
void main(){}
"""


class MeshGL:
    def __init__(self, gl, V, N, F):
        self.gl = gl
        ctx = gl.ctx
        if not hasattr(gl, "_mesh_prog"):
            gl._mesh_prog = ctx.program(vertex_shader=VS_MESH, fragment_shader=FS_MESH)
            gl._depth_prog = ctx.program(vertex_shader=VS_DEPTH, fragment_shader=FS_DEPTH)
        self.prog, self.dprog = gl._mesh_prog, gl._depth_prog
        data = np.c_[V, N].astype("f4")
        self.vbo = ctx.buffer(data.tobytes())
        self.ibo = ctx.buffer(np.asarray(F, "i4").tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 3f", "in_pos", "in_nrm")], self.ibo)
        self.dvao = ctx.vertex_array(self.dprog, [(self.vbo, "3f 12x", "in_pos")], self.ibo)

    def draw_depth(self, light_vp, model=np.eye(4)):
        self.dprog["u_light_vp"].write(np.asarray(light_vp, "f4").T.tobytes())
        self.dprog["u_model"].write(np.asarray(model, "f4").T.tobytes())
        self.dvao.render(moderngl.TRIANGLES)

    def draw(self, vp, cam, model=np.eye(4), light_vp=np.eye(4), shadow_tex=None, **u):
        p = self.prog
        p["u_vp"].write(np.asarray(vp, "f4").T.tobytes())
        p["u_model"].write(np.asarray(model, "f4").T.tobytes())
        p["u_light_vp"].write(np.asarray(light_vp, "f4").T.tobytes())
        p["u_cam"].value = tuple(map(float, cam))
        defaults = dict(ldir=(-0.3, -0.6, -0.4), lcol=(1, 1, 1), amb_top=(0.2, 0.22, 0.25), amb_bot=(0.05, 0.05, 0.05),
                        albedo=(0.5, 0.5, 0.5), spec=0.3, rough=0.5, rim=0.0, fog_col=(0, 0, 0), fog_density=0.0,
                        fog_height=500.0, emit=(0, 0, 0), alpha=1.0)
        defaults.update(u)
        for k, v in defaults.items():
            if "u_" + k in p:
                p["u_" + k].value = tuple(map(float, v)) if isinstance(v, (tuple, list, np.ndarray)) else float(v)
        if shadow_tex is not None:
            shadow_tex.use(5)
            p["u_shadow"].value = 5
            p["u_use_shadow"].value = 1
        else:
            p["u_use_shadow"].value = 0
        self.vao.render(moderngl.TRIANGLES)


def ortho(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return m


class ShadowMap:
    def __init__(self, gl, size=2048):
        ctx = gl.ctx
        self.size = size
        self.tex = ctx.depth_texture((size, size))
        self.tex.compare_func = ""
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = ctx.framebuffer(depth_attachment=self.tex)

    def begin(self):
        self.fbo.use()
        self.fbo.clear(depth=1.0)
