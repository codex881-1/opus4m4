#!/usr/bin/env bash
# Fetch open-licence fonts (SIL OFL) and Natural Earth vectors (public domain).
# DEM tiles (AWS Terrain Tiles, terrarium) are downloaded on demand by engine/geo.py.
# CJK glyphs come from the system package fonts-noto-cjk.
set -euo pipefail
cd "$(dirname "$0")/../assets"
mkdir -p fonts geo dem

G=https://raw.githubusercontent.com/google/fonts/main/ofl
for f in \
  ibmplexmono/IBMPlexMono-Light.ttf ibmplexmono/IBMPlexMono-Regular.ttf ibmplexmono/IBMPlexMono-Medium.ttf \
  ibmplexmono/IBMPlexMono-SemiBold.ttf ibmplexmono/OFL.txt \
  ibmplexsanscondensed/IBMPlexSansCondensed-Light.ttf ibmplexsanscondensed/IBMPlexSansCondensed-Regular.ttf \
  ibmplexsanscondensed/IBMPlexSansCondensed-Medium.ttf ibmplexsanscondensed/IBMPlexSansCondensed-SemiBold.ttf \
  ibmplexserif/IBMPlexSerif-Light.ttf ibmplexserif/IBMPlexSerif-Regular.ttf ibmplexserif/IBMPlexSerif-Italic.ttf \
  ibmplexserif/IBMPlexSerif-Medium.ttf instrumentserif/InstrumentSerif-Regular.ttf \
  instrumentserif/InstrumentSerif-Italic.ttf courierprime/CourierPrime-Regular.ttf courierprime/CourierPrime-Bold.ttf; do
  [ -f "fonts/$(basename "$f")" ] || curl -sSfL -m 120 -o "fonts/$(basename "$f")" "$G/$f"
done

N=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson
for f in ne_10m_coastline ne_10m_land ne_10m_admin_0_boundary_lines_land ne_10m_admin_1_states_provinces_lines \
         ne_10m_rivers_lake_centerlines ne_10m_lakes ne_10m_populated_places_simple ne_50m_admin_1_states_provinces \
         ne_10m_ocean; do
  [ -f "geo/$f.geojson" ] || curl -sSfL -m 300 -o "geo/$f.geojson" "$N/$f.geojson"
done
echo "assets ready"
