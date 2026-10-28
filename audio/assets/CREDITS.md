# Credits

## Sound recordings
All 22 recordings in the mix are from Freesound under **CC0 1.0**; no attribution is required, but credit is given here. Per-file metadata is in `rec/<id>.json`, and `fetch_recordings.py` re-downloads them.

| ID | Title | Author | Licence | Source |
|---|---|---|---|---|
| 31021 | Proyector16mm-2.wav | espectral | CC0 1.0 | https://freesound.org/people/espectral/sounds/31021/ |
| 132861 | Distant Explosion.wav | ecfike | CC0 1.0 | https://freesound.org/people/ecfike/sounds/132861/ |
| 154403 | paper slide.wav | semccab | CC0 1.0 | https://freesound.org/people/semccab/sounds/154403/ |
| 170548 | Longest Sound Typewriter | fellwell5 | CC0 1.0 | https://freesound.org/people/fellwell5/sounds/170548/ |
| 183656 | 50 Caliber MG.flac | qubodup | CC0 1.0 | https://freesound.org/people/qubodup/sounds/183656/ |
| 221899 | B-25 FLYOVER.wav | tubbers | CC0 1.0 | https://freesound.org/people/tubbers/sounds/221899/ |
| 253090 | Footsteps over Wooden Bridge (WAV) | OBXJohn | CC0 1.0 | https://freesound.org/people/OBXJohn/sounds/253090/ |
| 399926 | Carabiner Attach to Harness | Kinoton | CC0 1.0 | https://freesound.org/people/Kinoton/sounds/399926/ |
| 422513 | Swinging staff whoosh (strong) 04.wav | Nightflame | CC0 1.0 | https://freesound.org/people/Nightflame/sounds/422513/ |
| 437835 | G06-16_C-47 2 Engine Prop Int Start and Idle.wav | craigsmith | CC0 1.0 | https://freesound.org/people/craigsmith/sounds/437835/ |
| 448474 | es-stamp.wav | eddies2000 | CC0 1.0 | https://freesound.org/people/eddies2000/sounds/448474/ |
| 486557 | thunder.rumble.ogg | xUMR | CC0 1.0 | https://freesound.org/people/xUMR/sounds/486557/ |
| 528259 | Prison door closing | brianbaltar | CC0 1.0 | https://freesound.org/people/brianbaltar/sounds/528259/ |
| 552160 | Radio tuning-static-interference | quantumriver | CC0 1.0 | https://freesound.org/people/quantumriver/sounds/552160/ |
| 574566 | Wind winter strong cold blustery muffled gusts 77mel 190120 .mp3 | TRP | CC0 1.0 | https://freesound.org/people/TRP/sounds/574566/ |
| 575245 | Wind, winter, strong, trees, cold, 08.mp3 | TRP | CC0 1.0 | https://freesound.org/people/TRP/sounds/575245/ |
| 591873 | Train Horn 71 (Distant, Clean).mp3 | FunWithSound | CC0 1.0 | https://freesound.org/people/FunWithSound/sounds/591873/ |
| 643500 | Key Steel Vault Lock Turn | fractionalist | CC0 1.0 | https://freesound.org/people/fractionalist/sounds/643500/ |
| 683773 | Water Dripping on Concrete (off mic) | Elements-Library | CC0 1.0 | https://freesound.org/people/Elements-Library/sounds/683773/ |
| 687948 | Signal Electric Mfg. Co. R-68 Morse Practice Buzzer | scruss | CC0 1.0 | https://freesound.org/people/scruss/sounds/687948/ |
| 794818 | SWSH_Swish Of Heavy Stick.Artificial 1_EM | newlocknew | CC0 1.0 | https://freesound.org/people/newlocknew/sounds/794818/ |
| 812798 | Early Morning | Ultra-Edward | CC0 1.0 | https://freesound.org/people/Ultra-Edward/sounds/812798/ |

## Synthesised accents
Impacts, sub sink, tape fall, clicks and ticks were rendered with [earshot](https://github.com/Leonxlnx/earshot) 0.4.0 (MIT): pure DSP, no samples. Seeds are in `synth/render.sh`.

## Narration
Female narrator: [Kyutai Pocket TTS](https://kyutai.org/pocket-tts-technical-report/) (package code MIT), built-in voice `jane` (`kyutai/tts-voices/vctk/p339_023_enhanced.wav`, derived from the VCTK corpus, CC BY 4.0). The weights licence was not verified in this repo. Check the model card before public distribution.

## Data and type
- Elevation: Terrain Tiles on AWS (Mapzen/Tilezen terrarium; sources include SRTM, GMTED2010, ETOPO1).
- Vectors: Natural Earth (public domain).
- Fonts: IBM Plex Mono / Sans Condensed / Serif, Instrument Serif, Courier Prime (SIL OFL 1.1); Noto Serif CJK (SIL OFL 1.1).
- Moon position for 29 Nov 1952 computed with PyEphem.
