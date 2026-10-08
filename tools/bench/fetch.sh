#!/usr/bin/env bash
# Fetch the CC0 clip library for tools/bench/bench.py (outside the repo).
#   Quaternius Universal Animation Library 1 + 2 [Standard], CC0 1.0
#   (itch.io free downloads; License.txt in each zip). Checksums match the
#   HLL provenance log (~/SWE/games/tools/asset-pipeline.md).
# Bodies come from weightforge's corpus: ~/SWE/blender/weightforge/bench/corpus/fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FB="${FORGE_BENCH:-$HOME/.cache/forge-bench}"
mkdir -p "$FB/dl" "$FB/clips/ual2"
get() { # page upload-name file sha
  if [ ! -f "$FB/dl/$3" ] || ! echo "$4  $FB/dl/$3" | sha256sum -c --quiet - 2>/dev/null; then
    python3 "$HERE/itch_fetch.py" "$1" "$2" "$FB/dl/$3" "$4"
  fi
}
get https://quaternius.itch.io/universal-animation-library "[Standard]" ual1_standard.zip \
    cc73fc4e495b82958207316596317a3f40b9fa38065bde1027937452da537724
get https://quaternius.itch.io/universal-animation-library-2 "[Standard]" ual2_standard.zip \
    4008ea208a604773a2b2177d965f0f5d3195498b5bf838c3f5785d68e95f2a68
unzip -o -j -q "$FB/dl/ual1_standard.zip" "*/Unreal-Godot/UAL1_Standard_RM.glb" "*/License.txt" -d "$FB/clips"
unzip -o -j -q "$FB/dl/ual2_standard.zip" "*/Unreal-Godot/Mannequin_F.glb" -d "$FB/clips/ual2"
WF="${WEIGHTFORGE:-$HERE/../../../weightforge}"
if [ ! -d "$FB/corpus" ] && [ -x "$WF/bench/corpus/fetch.sh" ]; then "$WF/bench/corpus/fetch.sh"; fi
