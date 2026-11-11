#!/usr/bin/env bash
# Calibración de pista + detección en una sola orden, en lote.
#
#   ./scripts/calibrate_and_detect.sh                       # todos los vídeos de in/
#   ./scripts/calibrate_and_detect.sh noa_penalti           # uno solo, buscado en in/
#   ./scripts/calibrate_and_detect.sh in/noa_penalti.mp4 30 --force --open
#
# Entrada : in/<video>            (se crea si no existe)
# Salida  : output/<video>/       vídeo anotado + minimap.mp4 + heatmap.png + JSON
# Calib.  : config/calibration_<video>.yaml (se reutiliza si ya existe y es de ese vídeo)
#
# Por cada vídeo: calibra si hace falta (GUI de 4 clics o --points) y luego lanza el pipeline.
# Opciones:
#   --force             recalibrar aunque el YAML exista
#   --points "..."      calibración sin GUI: 'x1,y1 x2,y2 x3,y3 x4,y4' (píxeles del frame 0)
#   --calibration FILE  usar/crear ese YAML (p. ej. varios clips de la misma cámara)
#   --open              abrir los artefactos al terminar
set -euo pipefail
cd "$(dirname "$0")/.."

IN_DIR="in"
OUT_ROOT="output"
MODEL="runs/detect/train-3/weights/best.pt"
VIDEO_EXTENSIONS="mp4 mov m4v avi mkv webm MKV MOV MP4 AVI"
SECONDS_TO_PROCESS=20
FORCE=0
OPEN=0
POINTS=""
CALIB_OVERRIDE=""
POSITIONAL=()

usage() {
  cat <<'EOF'
uso: scripts/calibrate_and_detect.sh [video...] [segundos] [opciones]

  [video...]           vídeos de in/ (nombre o ruta). Sin argumentos: todos los de in/
  [segundos]           segundos a procesar por vídeo (def. 20)
  --force              recalibrar aunque exista config/calibration_<video>.yaml
  --points "..."       calibración sin GUI: 'x1,y1 x2,y2 x3,y3 x4,y4' del frame 0
  --calibration FILE   usar o crear ese YAML en vez de config/calibration_<video>.yaml
  --open               abrir los artefactos al terminar

Entrada: in/   Salida: output/<video>/
EOF
}

die() { echo "error: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --force) FORCE=1; shift ;;
    --open) OPEN=1; shift ;;
    --points)
      [ $# -ge 2 ] || die "--points necesita un valor"
      POINTS="$2"; shift 2 ;;
    --calibration)
      [ $# -ge 2 ] || die "--calibration necesita una ruta"
      CALIB_OVERRIDE="$2"; shift 2 ;;
    -*) die "opción desconocida: $1 (usa --help)" ;;
    *)
      case "$1" in
        ''|*[!0-9.]*|*.*.*) POSITIONAL+=("$1") ;;
        *) SECONDS_TO_PROCESS="$1" ;;
      esac
      shift ;;
  esac
done

mkdir -p "$IN_DIR" "$OUT_ROOT"

# --- Resolver la lista de vídeos ---------------------------------------------
VIDEOS=()
if [ ${#POSITIONAL[@]} -eq 0 ]; then
  shopt -s nullglob
  for extension in $VIDEO_EXTENSIONS; do
    for candidate in "$IN_DIR"/*."$extension"; do
      VIDEOS+=("$candidate")
    done
  done
  shopt -u nullglob
  if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "error: no hay vídeos en $IN_DIR/. Copia allí un .mp4 (o pásame la ruta)." >&2
    exit 1
  fi
else
  for name in "${POSITIONAL[@]}"; do
    if [ -f "$name" ]; then
      VIDEOS+=("$name")
    elif [ -f "$IN_DIR/$name" ]; then
      VIDEOS+=("$IN_DIR/$name")
    elif [ -f "$IN_DIR/$name.mp4" ]; then
      VIDEOS+=("$IN_DIR/$name.mp4")
    else
      echo "error: no encuentro '$name' (ni como ruta ni en $IN_DIR/)" >&2
      exit 1
    fi
  done
fi

# --- Calibración: reutilizar o crear ----------------------------------------
resolve_calibration() {
  local video="$1"
  local stem calibration calib_video calib_frame

  stem="$(basename "${video%.*}")"
  calibration="${CALIB_OVERRIDE:-config/calibration_${stem}.yaml}"

  if [ "$FORCE" -eq 0 ] && [ -f "$calibration" ]; then
    calib_video="$(sed -n 's/^video: *//p' "$calibration" | head -1 | sed 's/^["'\'']//; s/["'\'']$//')"
    calib_frame="$(sed -n 's/^frame: *//p' "$calibration" | head -1)"
    if [ "$(basename "$calib_video")" != "$(basename "$video")" ]; then
      echo "  aviso: $calibration es de '$calib_video', no de '$video'; se recalibra." >&2
    elif [ "${calib_frame:-0}" != "0" ]; then
      echo "  aviso: $calibration usa el frame ${calib_frame}; el pipeline exige frame 0. Se recalibra." >&2
    else
      echo "  calibración reutilizada → $calibration" >&2
      echo "  (usa --force para volver a calibrar)" >&2
      printf '%s' "$calibration"
      return 0
    fi
  fi

  echo "  calibrando: clic en las 4 esquinas del suelo (el orden da igual)." >&2
  echo "  Z deshace, R reinicia, ENTER guarda, ESC cancela." >&2
  local calib_args=(--video "$video" --width 40 --height 20 --out "$calibration")
  if [ -n "$POINTS" ]; then
    calib_args+=(--points "$POINTS")
  fi
  # La salida de calibrate.py va a stderr: stdout está reservado para la ruta del YAML.
  if ! uv run python scripts/calibrate.py "${calib_args[@]}" >&2; then
    return 1
  fi
  [ -f "$calibration" ] || return 1
  printf '%s' "$calibration"
}

run_pipeline() {
  local video="$1"
  local calibration="$2"
  local stem out
  local -a args

  stem="$(basename "${video%.*}")"
  out="$OUT_ROOT/$stem"

  args=(--video "$video" --calibration "$calibration" --out "$out"
        --max-seconds "$SECONDS_TO_PROCESS" --tile auto --imgsz 960)
  if [ -f "$MODEL" ]; then
    args+=(--model "$MODEL" --detector-classes "0,1,2")
  fi
  uv run python -m player_detect.cli "${args[@]}"

  echo
  echo "Artefactos en $out:"
  find "$out" -maxdepth 1 -type f -exec du -h {} + | sort -h | sed "s|$out/|  |"

  local missing=0
  if [ ! -f "$out/minimap.mp4" ]; then
    echo "  aviso: no hay minimap.mp4 (calibración sin proyecciones válidas)" >&2
    missing=1
  fi
  if [ ! -f "$out/heatmap.png" ]; then
    echo "  aviso: no hay heatmap.png (ninguna detección proyectada a la pista)" >&2
    missing=1
  fi
  if [ "$missing" -eq 1 ]; then
    echo "  revisa la calibración y el aviso de cobertura del resumen de arriba" >&2
  fi

  if [ "$OPEN" -eq 1 ]; then
    local artifact
    for artifact in "$out"/*_annotated.mp4 "$out/minimap.mp4" "$out/heatmap.png"; do
      [ -f "$artifact" ] && open "$artifact"
    done
  fi
  return "$missing"
}

# --- Bucle principal ---------------------------------------------------------
if [ ! -f "$MODEL" ]; then
  echo "aviso: no hay modelo propio en $MODEL; se usa el por defecto (solo 'person')" >&2
fi

total=${#VIDEOS[@]}
done_ok=0
failed=0
warned=0

for index in "${!VIDEOS[@]}"; do
  video="${VIDEOS[$index]}"
  echo
  echo "=== [$((index + 1))/$total] $video (${SECONDS_TO_PROCESS}s) ==="

  if ! calibration="$(resolve_calibration "$video")"; then
    echo "  error: calibración cancelada o inválida; se salta este vídeo" >&2
    failed=$((failed + 1))
    continue
  fi

  echo "  calibración: $calibration"
  echo "  procesando → $OUT_ROOT/$(basename "${video%.*}")"
  if run_pipeline "$video" "$calibration"; then
    done_ok=$((done_ok + 1))
  else
    warned=$((warned + 1))
  fi
done

# --- Resumen global ----------------------------------------------------------
echo
if [ "$total" -gt 1 ]; then
  echo "Resumen: $done_ok completado(s), $warned con avisos, $failed fallido(s) de $total."
fi
if [ "$failed" -gt 0 ]; then
  exit 1
fi
