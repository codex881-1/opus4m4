#!/usr/bin/env bash
# Provenance of the synthesised accents (earshot 0.4.0, MIT; pure DSP, no samples).
# The committed WAVs are the exact takes used; seeds below reproduce them.
set -euo pipefail
cd "$(dirname "$0")"
E="${EARSHOT_PY:-python} -m earshot sfx render"
$E impact/deep          -o impact_deep.wav  --seed 0
$E impact/swell         -o impact_swell.wav --seed 0
$E impact/punch         -o impact_punch.wav --seed 0
$E downlifter/sub_sink  -o sub_sink.wav     --seed 579362556
$E downlifter/tape      -o tape_fall.wav    --seed 661058652
$E click/thock          -o click_thock.wav  --seed 992592500
$E click/soft           -o click_soft.wav   --seed 76560324
$E hover_tick/detent    -o tick_detent.wav  --seed 700743171
$E knock/desk           -o knock_desk.wav   --seed 1
$E fit/snap/switch_flip -o switch_flip.wav  --seed 0
$E fit/tick/domino      -o tick_domino.wav  --seed 0
