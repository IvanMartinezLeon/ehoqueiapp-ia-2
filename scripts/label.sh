#!/usr/bin/env bash
# Launch labelImg with the project dataset preconfigured.
# Usage: scripts/label.sh [folder-de-imagenes]
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES="${1:-dataset/images/raw}"
CLASSES="dataset/classes.txt"
LABELS="dataset/labels/raw"

mkdir -p "$LABELS"
if [ ! -x .venv/bin/labelImg ]; then
  echo "labelImg no está instalado. Ejecuta: uv sync --group label" >&2
  exit 1
fi
# labelImg 1.8.6 crashes with PyQt5 >= 5.15.11 (float coordinates); patch it once.
if [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/patch_labelimg.py >/dev/null 2>&1 || true
fi
exec .venv/bin/labelImg "$IMAGES" "$CLASSES" "$LABELS"
