#!/usr/bin/env bash
# Atajos para los vídeos de ejemplo.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-help}" in
  okliga)
    shift || true
    uv run python -m player_detect.cli --video videos/okliga_prueba.mp4 --max-seconds "${1:-20}" --imgsz 640
    ;;
  palau)
    shift || true
    uv run python -m player_detect.cli --video videos/palau.mp4 --max-seconds "${1:-15}" --imgsz 960 --tile 2
    ;;
  *)
    echo "uso: scripts/run.sh {okliga|palau} [segundos]"
    exit 1
    ;;
esac
