# Image-generation prompts: "The Pickup"

Everything in the finished film is made in code: real elevation data, a parametric C-47, procedural paper, and vector or GL graphics. These prompts cover four inserts that code can't render convincingly: photoreal close-ups of hands, faces, fabric and props. They're optional. Each prompt is self-contained for a diffusion model (Midjourney v7 / FLUX.2 / Imagen 4 class). Keep the film's grade: deep blue-black shadows, warm ivory paper highlights, one amber accent, a desaturated 1950s palette, fine 35 mm grain.

House style suffix (append to every prompt):
> shot on 35mm Kodak Double-X pushed one stop, low-key practical lighting, 2.39:1 framing, shallow depth of field, slight halation on highlights, muted cold palette with a single warm practical, no text, no logos, no modern objects, documentary realism, not stylized, not illustrated

Negative prompt (all): `text, letters, watermark, logo, modern clothing, modern aircraft, smartphones, cartoon, CGI look, oversaturated, lens flare streaks, extra fingers, deformed hands, airbrushed skin`

---

## 1. Winch station, C-47 cargo door, night (inserts before the catch, ~0:23)
Interior of a 1950s olive-drab Douglas C-47 cargo hold at night, seen from just behind two young men in heavy 1950s winter flight clothing: sheepskin-collared leather jackets, wool caps, and canvas safety harnesses clipped to the fuselage frame. Neither face is visible. They lean toward an open rear cargo door. A manual steel cable winch with a hand crank is bolted to the floor. A taut steel cable runs out of the door into darkness, vibrating. The only light is pale blue moonlight through the door and a dim red cockpit glow far forward. Riveted aluminium ribs, cargo tie-down rings, frost on the door frame. Motion blur on the cable. Tense, quiet, 35 mm documentary still.

## 2. The pickup zone from the ground, moonlit snow (alternative plate for the ambush, ~0:33)
Wide night landscape in the Changbai Mountains, Jilin, China, winter 1952. A small clearing in dense snow-covered conifer forest on rolling hills under a 95% waxing gibbous moon high in the sky. Two thin 15-foot wooden poles stand about 50 feet apart with a rope sagging between them. Three small signal fires burn in a line in the snow. Four or five tiny dark figures in padded winter coats stand near the poles; one wears a parachute-style harness. At the edges of the clearing, two low shapes are covered by white sheets on the snow. Far away, the dark silhouette of a twin-engine propeller transport is approaching low, with faint orange exhaust glow. Blue moonlight, long soft shadows, cold breath haze, stars visible. Photoreal, wide-angle 28 mm, deep focus.

## 3. Cell detail, single bulb (alternative plate for the days section, ~0:52)
A tiny 5-by-8-foot prison cell with four bare cement walls, a heavy dark wooden door with a small eyehole and a sliding wooden cover, and a whitewashed window too high to see out of. A straw-filled mattress lies on the floor. A single bare incandescent bulb hangs from a cord and burns in the middle of the ceiling, its filament visible, with light falling off sharply into the corners. Nobody is in the cell. Dust in the air catches the light. Cold, quiet, symmetrical composition, eye-level 35 mm lens, deep shadows, faint warm cast from the bulb against cold grey cement.

## 4. Lo Wu crossing at dawn, 12 March 1973 (alternative plate for the bridge, ~1:01)
Early-morning fog over the Shenzhen River at the Lo Wu border crossing between mainland China and British Hong Kong, 1973. A long steel railway truss bridge recedes into mist. On its wooden pedestrian walkway, a single middle-aged man in a dark coat walks toward camera, small in frame, carrying nothing. On the near side, a British soldier in 1970s olive uniform and beret stands at attention, saluting, seen from behind. Soft diffused dawn light, pale grey-green palette, distant hills dissolving in haze. Telephoto 135 mm compression, quiet and dignified, a photojournalistic still.

---

### Notes for compositing
- Generate at 3072×1286 (2.39:1) or larger. The film letterboxes those sequences at 0.115 of frame height per bar.
- Grade inserts to match: lift shadows toward (0.010, 0.012, 0.016), split-tone cool shadows and warm highlights, then add 4.5% luminance-weighted grain. `engine/gl.py → FS_POST` reproduces the exact grade if inserts are routed through `GL.post()`.
- Do not generate portraits of Downey or Fecteau. Their likenesses are real people's; use archival photos (CIA.gov public-domain releases) with attribution if faces are needed.
