#!/usr/bin/env bash
# Demo de una sola orden: vídeo → vídeo anotado + minimapa + heatmap + JSON.
#
#   ./scripts/demo.sh                      # 20 s del clip de referencia (~1 min de proceso)
#   ./scripts/demo.sh videos/mi_video.mp4 30
#   ./scripts/demo.sh videos/mi_video.mp4 30 --open
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${1:-videos/okliga_prueba.mp4}"
SECONDS_TO_PROCESS="${2:-20}"
OPEN_ARTIFACTS="${3:-}"
OUT="runs/demo_$(basename "${VIDEO%.*}")"
MODEL="runs/detect/train-3/weights/best.pt"
CALIBRATION="config/calibration_$(basename "${VIDEO%.*}").yaml"

if [ ! -f "$VIDEO" ]; then
  echo "error: no existe el vídeo $VIDEO" >&2
  exit 1
fi

ARGS=(--video "$VIDEO" --max-seconds "$SECONDS_TO_PROCESS" --imgsz 640 --out "$OUT")
if [ -f "$MODEL" ]; then
  ARGS+=(--model "$MODEL" --detector-classes "0,1,2")
else
  echo "aviso: no hay modelo propio en $MODEL; se usa el por defecto (solo personas)" >&2
fi
if [ -f "$CALIBRATION" ]; then
  ARGS+=(--calibration "$CALIBRATION")
else
  echo "aviso: sin calibración para este vídeo; no habrá minimapa." >&2
  echo "       créala con: uv run python scripts/calibrate.py --video $VIDEO" >&2
fi

echo "Procesando ${SECONDS_TO_PROCESS}s de $VIDEO → $OUT"
uv run python -m player_detect.cli "${ARGS[@]}"

echo
echo "Artefactos generados en $OUT:"
find "$OUT" -maxdepth 1 -type f -exec du -h {} + | sort -h | sed "s|$OUT/|  |"

if [ "$OPEN_ARTIFACTS" = "--open" ]; then
  open "$OUT"/*_annotated.mp4 2>/dev/null || true
  [ -f "$OUT/minimap.mp4" ] && open "$OUT/minimap.mp4"
  [ -f "$OUT/heatmap.png" ] && open "$OUT/heatmap.png"
fi
