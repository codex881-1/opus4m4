"""Master edit decision list. Scene starts and beats are locked to narration word timestamps (audio/vo/timing.json)."""
import json
import os
from scenes.base import Timeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMING = os.path.join(ROOT, "audio", "vo", "timing.json")
TOTAL = 69.9


def word_time(words, w, n=1):
    k = 0
    for x in words:
        if x["w"] == w:
            k += 1
            if k == n:
                return x["t0"]
    raise KeyError(w)


def edl():
    words = json.load(open(TIMING))["words"]
    T = lambda w, n=1: word_time(words, w, n)
    s_map, s_bp, s_amb, s_jack, s_days, s_bridge = 7.2, 19.7, 31.3, 46.5, 51.3, 60.6
    return [
        ("letter", 0.0, s_map + 0.8, dict(type0=T("in") - 0.1, type1=T("japan") + 0.1), None),
        ("map", s_map, s_bp + 0.6 - s_map,
         dict(cover=-0.7, strike=T("story") - s_map, real=T("john") - s_map, cards=T("downey") - s_map,
              zoom=T("job") - s_map - 0.3, site=T("snow") - s_map - 0.7),
         dict(dur=0.8, mode=1, soft=0.2)),
        ("blueprint", s_bp, s_amb + 0.5 - s_bp,
         dict(poles=T("poles") - s_bp - 0.25, plane=T("plane") - s_bp + 0.25, catch=T("catches") - s_bp + 0.2,
              winch=T("winches") - s_bp, rewind=T("but") - s_bp, trap=T("caught") - s_bp - 0.3,
              guns=T("radio") - s_bp + 0.3),
         dict(dur=0.6, mode=1, soft=0.25)),
        ("ambush", s_amb, s_jack - s_amb,
         dict(guns=T("white") - s_amb - 0.05, hit=T("white") - s_amb + 0.55, crash=T("crashed") - s_amb + 0.05),
         dict(dur=0.5, mode=3)),
        ("jack", s_jack, s_days - s_jack, dict(quote=T("you") - s_jack - 0.05), None),
        ("days", s_days, s_bridge + 0.6 - s_days,
         dict(sweep0=0.3, sweep1=T("nixon") - s_days + 0.9, nixon=T("nixon") - s_days, hl0=T("downey", 4) - s_days),
         dict(dur=0.25, mode=3)),
        ("bridge", s_bridge, TOTAL - s_bridge,
         dict(border=T("kong") - s_bridge, final=T("7") - s_bridge, card=T("days", 1) - s_bridge + 1.1),
         dict(dur=0.6, mode=0)),
    ]


def build_timeline(gl, only=None):
    from scenes.s_letter import LetterScene
    from scenes.s_map import MapScene
    from scenes.s_blueprint import BlueprintScene
    from scenes.s_ambush import AmbushScene
    from scenes.s_flat import JackScene, DaysScene, BridgeScene
    classes = dict(letter=LetterScene, map=MapScene, blueprint=BlueprintScene, ambush=AmbushScene,
                   jack=JackScene, days=DaysScene, bridge=BridgeScene)
    tl = Timeline(gl)
    for name, start, dur, beats, trans in edl():
        if only and name not in only:
            continue
        sc = classes[name](gl, dur, beats=beats)
        tl.add(0.0 if only and len(only) == 1 else start, sc, trans)
    return tl
