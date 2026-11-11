# SPEC — Detección y seguimiento de jugadores en vídeo

Estado: **aprobada e implementada** (v1).

## 1. Objetivo

CLI en Python que, dado un vídeo de fútbol sala, detecta a todas las personas en pista
(jugadores/as y árbitros), les asigna un ID persistente en el tiempo y produce:

1. Un vídeo MP4 anotado (cajas + ID + confianza).
2. Un JSON con las detecciones por frame y un resumen por trayectoria.

## 2. Alcance

### Incluido (v1)

- Detección de la clase `person` (COCO) mediante YOLO11 de Ultralytics.
- Tracking multi-objeto con ByteTrack (BoT-SORT como alternativa configurable).
- Vídeo de entrada: cualquier formato soportado por ffmpeg.
- Vídeo de salida: MP4 H.264, misma resolución y fps que la entrada.
- Export `detections.json` (por frame) + `tracks.json` (resumen por ID).
- Modo `--tile` para vídeos de pista completa con jugadores muy pequeños.
- Selección de dispositivo automática: `mps` (Apple Silicon) → `cuda` → `cpu`.
- Procesado por lotes de frames (`--batch`) y límite de duración (`--max-seconds`) para pruebas rápidas.

### Fuera de alcance (v1)

- Clasificación por equipo (color de camiseta), lectura de dorsales.
- Homografía a coordenadas de pista, heatmaps, distancia recorrida, velocidad real.
- Estabilización de cámara / compensación de movimiento (se documenta la limitación).
- API, servicio web, frontend.
- Entrenamiento o fine-tuning de modelos.

## 3. Actores y casos de uso

| Caso | Entrada | Resultado esperado |
|---|---|---|
| Análisis rápido de clip | `videos/okliga_prueba.mp4` (60 s, 854x480, cámara con paneo) | MP4 anotado + JSON en < 2 min en M1 Pro |
| Partido completo, cámara fija | `videos/palau.mp4` (58 min, 640x360, gran angular) | MP4 anotado + JSON con `--tile` y `--imgsz 1280` |
| Prueba de humo | Cualquier vídeo con `--max-seconds 10` | Salida parcial para validar parámetros |

## 4. Requisitos funcionales

- **RF-1** El CLI acepta `--video` (obligatorio) y `--out` (por defecto `runs/<nombre_video>/`).
- **RF-2** Genera `<nombre>_annotated.mp4` con todas las detecciones cuyo `track_id` no sea nulo dibujadas.
- **RF-3** Genera `detections.json` con estructura:
  ```json
  {"video": "...", "fps": 25.0, "width": 854, "height": 480,
   "frames": [{"frame": 0, "t": 0.0,
               "detections": [{"track_id": 3, "bbox": [x1,y1,x2,y2], "conf": 0.91, "center": [cx,cy]}]}]}
  ```
- **RF-4** Genera `tracks.json` con, por `track_id`: número de frames, primer/último timestamp, longitud de la trayectoria en píxeles y bbox media.
- **RF-5** `--tile N` divide cada frame en una rejilla NxN con solape y fusiona detecciones con NMS (para `palau.mp4`).
- **RF-6** `--max-seconds S` limita el procesado a los primeros S segundos.
- **RF-7** Todos los parámetros (modelo, confianza, iou, imgsz, tracker, batch, device) configurables por flags y por `config/default.yaml`.
- **RF-8** Progreso visible en stderr con `tqdm` y resumen final (frames, tiempo, FPS medio, nº de trayectorias).

## 5. Requisitos no funcionales

- **RNF-1** Python 3.11 gestionado con `uv`; instalación reproducible vía `uv sync`.
- **RNF-2** Código tipado (`ruff` + `mypy` en modo laxo) y sin estado global.
- **RNF-3** Render con OpenCV; escritura de vídeo con `cv2.VideoWriter` + fallback a ffmpeg si falla el codec.
- **RNF-4** Nunca cargar el vídeo completo en memoria: streaming frame a frame.
- **RNF-5** CPU objetivo: Apple M1 Pro. `palau.mp4` completo en CPU es lento (>1 h): se documenta y se recomienda `--max-seconds` o GPU.

## 6. Criterios de aceptación

1. `uv run python -m player_detect.cli --video videos/okliga_prueba.mp4 --max-seconds 20` termina sin error y crea los 3 artefactos.
2. El MP4 anotado se reproduce correctamente en QuickTime/VLC y tiene la misma duración que el tramo procesado (± 1 frame).
3. En el clip de 20 s de `okliga_prueba.mp4` se detectan ≥ 5 trayectorias distintas y ningún ID se reutiliza de forma visible para dos personas simultáneas.
4. `detections.json` es JSON válido y parseable; `tracks.json` contiene tantas entradas como IDs distintos en `detections.json`.
5. `--tile 2 --imgsz 1280` sobre 10 s de `palau.mp4` produce más detecciones que el modo sin tiling.
6. `ruff check` y `pytest` pasan en verde.

## 7. Limitaciones conocidas

- Cámara en paneo (`okliga_prueba.mp4`): ByteTrack pierde IDs en movimientos bruscos y cambios de plano. Sin compensación de movimiento habrá saltos de ID; se mitiga con `track_buffer` alto.
- Gran angular (`palau.mp4`): jugadores de ~15 px sin tiling → recall bajo. El tiling aumenta el coste computacional de forma cuadrática.
- Árbitros y jugadores se detectan igual (misma clase `person`); separarlos requiere v2.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| Descarga inicial de pesos YOLO (requiere red) | Documentar; pesos se cachean en `~/.cache/ultralytics` |
| `torch` en M1 puede requerir versión específica | Fijar versiones en `pyproject.toml` |
| `VideoWriter` sin H.264 en algunos entornos | Fallback a ffmpeg vía subprocess |
| Vídeo de 58 min sin GPU | Flags `--max-seconds` / `--stride` (procesar 1 de cada N frames) |
