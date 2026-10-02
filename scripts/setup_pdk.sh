#!/usr/bin/env bash
# Fetch only the SKY130 1.8V core FET models (~10 MB of SPICE) and build
# ngspice-ready corner libraries under ./pdk_models.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/.pdk_src"
if [ ! -d "$SRC/.git" ]; then
  git clone -q --depth 1 --filter=blob:none --sparse \
    https://github.com/google/skywater-pdk-libs-sky130_fd_pr.git "$SRC"
  git -C "$SRC" sparse-checkout set cells/nfet_01v8 cells/pfet_01v8 models/parameters
fi
python3 "$ROOT/scripts/build_models.py" "$SRC" "$ROOT/pdk_models"
