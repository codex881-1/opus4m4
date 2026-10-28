"""Sound design + mix for 'The Pickup'. No music: recordings (Freesound CC0) + earshot-synthesised accents.

Every cue is placed from narration word times (audio/vo/timing.json) and the picture edit.
Output: audio/mix/soundtrack.wav (48 kHz stereo, mastered), stems in audio/mix/stems/.
"""
import json
import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "assets", "rec")
SYN = os.path.join(ROOT, "assets", "synth")
OUT = os.path.join(ROOT, "mix")
SR = 48000
TOTAL = 69.9
N = int(TOTAL * SR)
rng = np.random.default_rng(7)

timing = json.load(open(os.path.join(ROOT, "vo", "timing.json")))
WORDS = timing["words"]


def T(w, n=1):
    k = 0
    for x in WORDS:
        if x["w"] == w:
            k += 1
            if k == n:
                return x["t0"]
    raise KeyError(w)


# ------------------------------------------------------------------ io + dsp
_cache = {}


def load(name, folder=REC):
    key = (folder, name)
    if key not in _cache:
        y, sr = sf.read(os.path.join(folder, name), always_2d=True)
        if sr != SR:
            y = signal.resample_poly(y, SR, sr, axis=0)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, 1)
        _cache[key] = y[:, :2].astype(np.float64)
    return _cache[key]


def seg(y, a, b=None):
    a = int(a * SR)
    b = len(y) if b is None else int(b * SR)
    return y[a:b].copy()


def db(x):
    return 10 ** (x / 20)


def fades(y, fi=0.01, fo=0.05):
    n = len(y)
    i, o = min(int(fi * SR), n // 2), min(int(fo * SR), n // 2)
    if i:
        y[:i] *= np.linspace(0, 1, i)[:, None] ** 2
    if o:
        y[-o:] *= np.linspace(1, 0, o)[:, None] ** 2
    return y


def filt(y, kind, f, order=2):
    sos = signal.butter(order, f, kind, fs=SR, output="sos")
    return signal.sosfiltfilt(sos, y, axis=0)


def shelf(y, f, gain_db, kind="low"):
    # simple shelf via split + gain
    lo = filt(y, "low", f)
    hi = y - lo
    return (lo * db(gain_db) + hi) if kind == "low" else (lo + hi * db(gain_db))


def pan(y, p):
    """p in [-1, 1], constant power."""
    m = y.mean(1)
    a = (p + 1) * np.pi / 4
    return np.c_[m * np.cos(a), m * np.sin(a)] * np.sqrt(2)


def width(y, w):
    m, s = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
    return np.c_[m + s * w, m - s * w]


def make_ir(rt60, predelay=0.012, er=0.25, bright=6000, seed=1):
    r = np.random.default_rng(seed)
    n = int(rt60 * 1.2 * SR)
    t = np.arange(n) / SR
    env = np.exp(-6.91 * t / rt60)
    ir = r.standard_normal((n, 2)) * env[:, None]
    ir = filt(ir, "low", bright)
    pd = int(predelay * SR)
    ir = np.r_[np.zeros((pd, 2)), ir]
    for k in range(6):  # sparse early reflections
        d = int(r.uniform(0.004, 0.045) * SR) + pd
        ir[d] += r.uniform(-1, 1, 2) * er
    return ir / np.sqrt((ir ** 2).sum() / 2)


IRS = {"room": make_ir(0.45, 0.006, 0.4, 7000, 2), "concrete": make_ir(1.6, 0.018, 0.5, 5200, 3),
       "outdoor": make_ir(2.4, 0.03, 0.15, 3800, 4), "hall": make_ir(3.2, 0.03, 0.2, 4200, 5)}


def verb(y, kind, wet=0.3, dry=1.0):
    w = signal.fftconvolve(y, IRS[kind][: len(IRS[kind])], axes=0)[: len(y) + len(IRS[kind])]
    out = np.zeros_like(w)
    out[: len(y)] += y * dry
    return out + w * wet


def norm_peak(y, peak_db=-1.0):
    return y / (np.max(np.abs(y)) + 1e-12) * db(peak_db)


def rms_env(y, win=0.05):
    m = np.abs(y).mean(1) if y.ndim == 2 else np.abs(y)
    k = int(win * SR)
    return np.convolve(m, np.ones(k) / k, mode="same")


def peak_time(y, win=0.25):
    e = rms_env(y, win)
    return np.argmax(e) / SR


def onsets(y, thresh_db=-22, min_gap=0.08):
    e = rms_env(y, 0.008)
    e = e / (e.max() + 1e-12)
    th = db(thresh_db)
    above = e > th
    idx = np.where(above[1:] & ~above[:-1])[0]
    out, last = [], -1e9
    for i in idx:
        if (i - last) / SR > min_gap:
            out.append(i / SR)
            last = i
    return out


# ------------------------------------------------------------------ buses
BUS = {k: np.zeros((N + SR * 6, 2)) for k in ("amb", "fx", "lowfx")}


def place(bus, t, y, gain_db=0.0):
    a = int(round(t * SR))
    if a < 0:
        y = y[-a:]
        a = 0
    b = min(a + len(y), len(BUS[bus]))
    BUS[bus][a:b] += y[: b - a] * db(gain_db)


def bed(bus, t0, t1, y, gain_db, fi=0.8, fo=0.8):
    dur = t1 - t0
    if len(y) < dur * SR:
        reps = int(np.ceil(dur * SR / len(y))) + 1
        y = np.concatenate([y] * reps)
    y = fades(y[: int(dur * SR)].copy(), fi, fo)
    place(bus, t0, y, gain_db)


def synth(name):
    return load(name, SYN)


# ------------------------------------------------------------------ design
def design():
    # ----- S1 letter: lamp, archive room, typewriter under the typing
    room = filt(rng.standard_normal((int(9 * SR), 2)) * 0.02, "low", 900)
    bed("amb", 0.0, 7.9, room, -24, fi=1.2, fo=0.6)
    place("fx", 0.18, verb(synth("switch_flip.wav"), "room", 0.25), -14)
    tw = load("170548.wav")
    tw_on = onsets(tw, -18, 0.06)
    tseg = seg(tw, 1.2, 1.2 + 5.3)
    tseg = filt(tseg, "high", 140)
    place("fx", T("in") - 0.1, verb(fades(tseg, 0.02, 0.4), "room", 0.2), -19)
    place("fx", 6.25, verb(fades(seg(load("154403.wav"), 0.0, 1.3), 0.01, 0.3), "room", 0.2), -16)  # paper slide into the map
    # ----- S2 map: memory of the cabin drone, strike, radio at the zone
    cab = load("437835.wav")
    cseg = seg(cab, 60.0, 74.0)
    cseg = filt(filt(cseg, "low", 2400), "high", 60)
    bed("amb", T("john") - 0.6, 20.0, cseg, -13, fi=1.4, fo=1.0)
    strike = fades(filt(seg(load("794818.wav"), 0.0, 0.5), "high", 1500), 0.002, 0.15)
    place("fx", T("story") - 0.02, strike, -17)
    place("fx", T("story") + 0.08, synth("knock_desk.wav"), -20)
    rad = filt(seg(load("552160.wav"), 2.0, 7.0), "band", [300, 3400])
    place("fx", T("snow") - 0.8, fades(rad, 0.4, 1.2), -23)
    mk = filt(seg(load("687948.wav"), 0.5, 4.0), "band", [500, 2600])
    place("fx", T("snow") - 0.2, fades(mk, 0.05, 0.8), -29)
    for i in range(3):
        place("fx", T("snow") - 0.7 + i * 1.25, synth("tick_detent.wav") * np.linspace(1, 0, len(synth("tick_detent.wav")))[:, None], -24)
    # ----- S3 blueprint: rig, the pass, the catch, the winch, rewind, the trap
    air = fades(seg(load("422513.wav"), 0.0), 0.002, 0.1)
    place("fx", 19.55, verb(air, "room", 0.3), -18)
    clank = load("399926.wav")
    c_on = onsets(clank, -14, 0.25)
    def clank_hit(k):
        a = c_on[k % len(c_on)] if c_on else 0.5
        return fades(seg(clank, max(0, a - 0.01), a + 0.35), 0.003, 0.15)
    place("fx", T("poles") - 0.1, clank_hit(0), -19)
    place("fx", T("line") - 0.05, clank_hit(1), -21)
    fly = load("221899.wav")
    fp = peak_time(fly, 0.4)
    catch_t = 19.7 + (T("catches") - 19.7 + 0.2)
    fseg = seg(fly, max(0, fp - 2.2), fp + 2.6)
    place("fx", catch_t - 2.2, fades(filt(fseg, "high", 70), 0.6, 1.2), -13)
    place("fx", catch_t, verb(clank_hit(2), "room", 0.2), -10)
    place("lowfx", catch_t, synth("impact_punch.wav"), -17)
    place("fx", catch_t + 0.05, verb(air, "room", 0.25), -15)
    # winch: ratchet built from real clank transients, accelerating then settling
    wt = T("winches")
    t = wt
    k = 0
    while t < wt + 1.7:
        rate = 9 + 7 * np.sin(np.pi * (t - wt) / 1.7)
        y = clank_hit(3 + k) * (0.5 + 0.3 * rng.random())
        place("fx", t, filt(y, "high", 900), -26)
        t += 1.0 / rate
        k += 1
    # rewind
    rw = np.flipud(fseg[: int(1.2 * SR)].copy())
    place("fx", T("but") - 0.05, fades(filt(rw, "band", [200, 4000]), 0.2, 0.3), -22)
    place("fx", T("but") + 0.1, synth("tape_fall.wav"), -24)
    # trap: low sink + radio + crosshair locks
    place("lowfx", T("caught") - 0.15, synth("sub_sink.wav"), -15)
    place("fx", T("radio") - 0.1, fades(filt(seg(load("552160.wav"), 9.0, 11.2), "band", [400, 3000]), 0.05, 0.6), -22)
    place("fx", T("radio") + 0.35, synth("click_thock.wav"), -18)
    place("fx", T("radio") + 0.5, synth("click_thock.wav"), -18)
    place("lowfx", T("trap") - 0.05, synth("impact_swell.wav")[int(0.25 * SR):], -21)
    # ----- S4 ambush
    s4 = 31.3
    wind = load("575245.wav")
    wseg = width(filt(seg(wind, 50.0, 66.0), "high", 60), 1.3)
    bed("amb", s4, 46.6, wseg, -22, fi=0.6, fo=1.2)
    wind2 = filt(seg(load("574566.wav"), 104.5, 120.0), "low", 1800)
    bed("amb", T("crashed") + 0.4, 51.0, wind2, -24, fi=2.0, fo=1.5)
    # approach + pass (exterior). Pass peak lands in the tracking shot; engines cut after the hit.
    hit = T("white") + 0.55
    eng = seg(fly, max(0, fp - 5.5), fp + 1.5)
    eng = filt(eng, "high", 50)
    e_start = hit - 5.5 + 0.2
    env = np.ones(len(eng))
    cut = int((hit + 0.7 - e_start) * SR)
    if 0 < cut < len(env):
        f = int(0.35 * SR)
        env[cut:cut + f] = np.linspace(1, 0, min(f, len(env) - cut))
        env[cut + f:] = 0
    place("fx", e_start, fades(eng * env[:, None], 1.2, 0.05), -17)
    # sheets flap
    flap = fades(filt(seg(load("794818.wav"), 0.6, 1.4), "band", [150, 5000]), 0.01, 0.2)
    place("fx", T("white") - 0.05, verb(pan(flap, 0.5), "outdoor", 0.25), -16)
    place("fx", T("white") + 0.05, verb(pan(flap, -0.5), "outdoor", 0.25), -18)
    # two guns, crossfire (.50 cal, distance-processed), L and R
    mg = load("183656.wav")
    mg_on = onsets(mg, -12, 0.6)
    burst_src = [(a, a + 1.6) for a in mg_on[:6]] or [(0.5, 2.1)]
    def burst(i):
        a, b = burst_src[i % len(burst_src)]
        y = seg(mg, max(0, a - 0.02), b)
        y = filt(filt(y, "low", 3200), "high", 110)
        return fades(y, 0.003, 0.25)
    g0 = T("white") + 0.02
    sched = [(0.00, -0.65), (0.18, 0.7), (0.95, -0.6), (1.25, 0.65), (1.75, -0.7), (1.95, 0.6), (2.45, -0.55), (2.7, 0.65), (3.15, -0.6)]
    for i, (dt, pn) in enumerate(sched):
        y = verb(pan(burst(i), pn), "outdoor", 0.55, 0.85)
        under_vo = (g0 + dt) < T("guns") + 0.25
        place("fx", g0 + dt, y, (-24 if under_vo else -15) - i * 0.4)
    # hits on the airframe: short metal transients
    for i in range(5):
        place("fx", hit + 0.05 + i * 0.09 + rng.random() * 0.04, filt(clank_hit(5 + i), "high", 1200), -20)
    # crash: heavy, muffled, with a long rumble tail - no Hollywood explosion
    cr = T("crashed") + 0.05
    boom = filt(seg(load("132861.wav"), 0.0), "low", 900)
    place("lowfx", cr, verb(boom, "outdoor", 0.4), -20)
    place("lowfx", cr, synth("impact_deep.wav"), -20)
    place("amb", cr + 0.25, fades(filt(seg(load("486557.wav"), 2.0, 12.0), "low", 600), 0.3, 3.0), -27)
    # fire crackle (synth: sparse filtered impulses), distant
    n = int(7.0 * SR)
    cr_sig = np.zeros((n, 2))
    for _ in range(420):
        i = rng.integers(0, n - 200)
        l = rng.integers(20, 160)
        cr_sig[i:i + l] += rng.standard_normal((l, 2)) * np.exp(-np.arange(l) / (l / 4))[:, None] * rng.random() ** 2
    cr_sig = filt(filt(cr_sig, "band", [700, 6000]), "low", 4500)
    bed("amb", cr + 0.6, 46.4, verb(cr_sig, "outdoor", 0.3), -30, fi=1.5, fo=1.0)
    # ----- S5 'You are Jack.': near silence, a lock, a drip, then nothing under the quote
    s5 = 46.5
    place("fx", s5 + 0.15, verb(fades(seg(load("643500.wav"), 0.2, 2.2), 0.01, 0.4), "concrete", 0.35), -17)
    drip = load("683773.wav")
    d_on = onsets(drip, -16, 0.4)
    if d_on:
        place("fx", s5 + 1.6, verb(fades(seg(drip, d_on[0] - 0.01, d_on[0] + 0.6), 0.003, 0.3), "concrete", 0.5), -20)
    rt = filt(rng.standard_normal((int(5 * SR), 2)) * 0.02, "low", 400)
    bed("amb", s5, 51.4, rt, -30, fi=0.4, fo=0.3)
    # ----- S6 days: the door, the bulb, the count, the transcript
    s6 = 51.3
    door = load("528259.wav")
    dpk = onsets(door, -10, 0.5)
    da = dpk[0] if dpk else 0.0
    place("fx", s6 + 0.02, verb(fades(seg(door, max(0, da - 0.03), da + 3.5), 0.005, 1.0), "concrete", 0.45), -12)
    place("lowfx", s6 + 0.02, synth("impact_punch.wav"), -22)
    tt = np.arange(int(9.6 * SR)) / SR
    fil = rng.standard_normal((len(tt), 2))
    fil *= (0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 50 * tt)))[:, None]   # mains-modulated filament noise, atonal
    buzz = filt(fil, "band", [1800, 7000]) * 0.4 + filt(rng.standard_normal((len(tt), 2)), "low", 300) * 0.6
    bed("amb", s6 + 0.3, 60.9, buzz, -44, fi=1.0, fo=0.8)
    # the count: soft mechanical ticks that accelerate with the years then stop on the Nixon date
    sw0, sw1 = s6 + 0.3, T("nixon") + 0.9
    t = sw0
    while t < sw1:
        u = (t - sw0) / (sw1 - sw0)
        place("fx", t, synth("tick_domino.wav") * (0.6 + 0.4 * rng.random()), -31 + 3 * np.sin(np.pi * u))
        t += (0.26 - 0.14 * np.sin(np.pi * u)) * (0.55 + 0.9 * rng.random())
    place("fx", T("nixon") - 0.05, verb(fades(seg(load("154403.wav"), 0.0, 1.2), 0.01, 0.3), "room", 0.2), -14)
    place("fx", T("downey", 4) + 0.05, fades(filt(seg(load("794818.wav"), 0.0, 0.45), "high", 2500), 0.002, 0.12), -22)
    st = load("448474.wav")
    s_on = onsets(st, -10, 0.3)
    sa = s_on[0] if s_on else 0.0
    place("fx", T("agent", 2) + 0.35, verb(fades(seg(st, max(0, sa - 0.01), sa + 0.5), 0.003, 0.2), "room", 0.2), -16)
    # ----- S7 Lo Wu: morning, river, distant train, boots on planks, the number
    s7 = 60.6
    morn = load("812798.wav")
    bed("amb", s7, 69.9, filt(seg(morn, 10.0, 22.0), "high", 120), -20, fi=1.0, fo=2.5)
    horn = load("591873.wav")
    hp = onsets(horn, -14, 1.0)
    ha = hp[0] if hp else 0.0
    place("amb", s7 + 0.9, verb(filt(fades(seg(horn, max(0, ha - 0.1), ha + 4.5), 0.2, 1.5), "low", 2500), "outdoor", 0.4), -26)
    steps = load("253090.wav")
    st_on = onsets(steps, -16, 0.3)
    walk_t = s7 + 0.5
    for i in range(min(9, len(st_on))):
        a = st_on[i]
        y = fades(seg(steps, max(0, a - 0.01), a + 0.35), 0.003, 0.15)
        place("fx", walk_t + i * 0.42, filt(y, "low", 5000), -24 - (i % 2))
    place("lowfx", T("7") - 0.02, synth("impact_deep.wav"), -15)
    place("fx", T("7") - 0.02, verb(synth("knock_desk.wav"), "hall", 0.5), -22)
    # end card: projector runs out -> silence
    proj = load("31021.wav")
    place("amb", T("days", 1) + 1.0, fades(filt(seg(proj, 3.0, 6.0), "band", [120, 6000]), 0.3, 1.6), -30)


def master():
    vo, sr = sf.read(os.path.join(ROOT, "vo", "narration.wav"))
    vo = np.c_[vo, vo][:N]
    vo = np.pad(vo, ((0, N - len(vo)), (0, 0)))
    env = rms_env(vo, 0.12)
    env = env / (env.max() + 1e-9)
    duck = np.clip(env * 3.0, 0, 1)
    duck = np.convolve(duck, np.ones(int(0.25 * SR)) / int(0.25 * SR), mode="same")
    ride = np.ones(N)
    for t0, t1, gdb in ((T("crashed") - 0.45, T("trees") + 0.55, -9.0),
                        (T("two", 2) - 0.1, T("guns") + 0.35, -4.0)):
        r = np.ones(N)
        a0, a1 = int(t0 * SR), int(t1 * SR)
        k = int(0.18 * SR)
        r[a0:a1] = db(gdb)
        r[a0 - k:a0] = np.linspace(1, db(gdb), k)
        r[a1:a1 + k] = np.linspace(db(gdb), 1, k)
        ride *= r
    BUS["fx"][:N] *= ride[:, None]
    BUS["amb"][:N] *= ride[:, None]
    BUS["lowfx"][:N] *= np.sqrt(ride)[:, None]
    amb = BUS["amb"][:N] * (1 - 0.55 * duck)[:, None]
    fx = BUS["fx"][:N] * (1 - 0.55 * duck)[:, None]
    low = BUS["lowfx"][:N] * (1 - 0.35 * duck)[:, None]
    low = filt(low, "high", 28)
    meter = pyln.Meter(SR)
    vo_l = meter.integrated_loudness(vo)
    vo = vo * db(-17.0 - vo_l)  # dialogue anchored at about -17 LUFS
    fx_bus = amb + fx + low
    mix = vo + fx_bus
    os.makedirs(os.path.join(OUT, "stems"), exist_ok=True)
    for name, y in (("vo", vo), ("amb", amb), ("fx", fx), ("low", low)):
        sf.write(os.path.join(OUT, "stems", name + ".wav"), y.astype(np.float32), SR, subtype="FLOAT")
    L = meter.integrated_loudness(mix)
    mix = mix * db(-15.0 - L)
    # true-peak-ish limiter: 4x oversampled peak detection, smooth gain
    up = signal.resample_poly(mix, 4, 1, axis=0)
    pk = np.abs(up).max(1).reshape(-1, 4).max(1)[: len(mix)]
    ceiling = db(-1.5)
    g = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    k = int(0.004 * SR)
    g = -np.convolve(-g, np.ones(k) / k, mode="same")
    g = np.minimum(g, ceiling / np.maximum(pk, 1e-9))
    rel = np.copy(g)
    a = np.exp(-1 / (0.08 * SR))
    for i in range(1, len(rel)):
        rel[i] = min(g[i], a * rel[i - 1] + (1 - a) * g[i])
    mix = mix * rel[:, None]
    fo = int(0.6 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
    sf.write(os.path.join(OUT, "soundtrack.wav"), mix.astype(np.float32), SR, subtype="FLOAT")
    Lf = meter.integrated_loudness(mix)
    up = signal.resample_poly(mix, 4, 1, axis=0)
    print(f"integrated {Lf:.1f} LUFS, true peak ~{20 * np.log10(np.abs(up).max()):.2f} dBTP, VO anchor {vo_l:.1f}->-17")


if __name__ == "__main__":
    design()
    master()
