# AGENTS.md

Guía para agentes (y humanos) que trabajan en este repositorio.

## Qué es este proyecto

Detección y seguimiento de jugadores en vídeos de hockey sobre patines / fútbol sala con YOLO + tracking.
Produce vídeo anotado, JSON de detecciones/trayectorias, minimapa cenital con coordenadas de pista y
heatmap por equipo. Incluye además herramientas para construir y entrenar un dataset YOLO propio.

## Comandos

```bash
uv sync                      # instalar dependencias (Python 3.11)
uv sync --group label        # + labelImg para etiquetar

uv run pytest                # 68 tests, sin red ni pesos
uv run ruff check src/ tests/ scripts/
uv run mypy src/

# pipeline principal
uv run python -m player_detect.cli --video videos/okliga_prueba.mp4 --max-seconds 20
uv run python -m player_detect.cli --video videos/okliga_prueba.mp4 \
  --model runs/detect/train-3/weights/best.pt --detector-classes 0,1,2 \
  --calibration config/calibration_okliga_prueba.yaml
```

## Mapa de módulos

| Fichero | Responsabilidad |
|---|---|
| `src/player_detect/config.py` | `Settings` inmutable (frozen+slots), YAML, validación, merge flags > yaml |
| `video.py` | lectura streaming (`VideoSource`) y escritura MP4 con reencode H.264 (`VideoSink`) |
| `detector.py` | YOLO + tiling NxN (`--tile`) para jugadores pequeños |
| `tracking.py` | ByteTrack / BoT-SORT / OCSort sobre `ultralytics.trackers` |
| `teams.py` | clasificación por color de camiseta (muestreo torso + k-means k=3, etiquetas sticky) |
| `camera.py` | homografía de cámara: `CameraTracker` (LK), `AnchorTracker` (frame 0), `compose`, `to_reference` |
| `world.py` | **v4**: `WorldTracker`, anclaje a keyframes verificado por `line_residual` |
| `pitch.py` | `Calibration` (4 puntos → metros), `order_quad`, `RinkProjector`, `foot_point` |
| `minimap.py` | lienzo cenital, puntos, `overlay`, `HeatmapAccumulator` |
| `draw.py` / `export.py` / `pipeline.py` / `cli.py` | render, JSON, orquestación, CLI |

Scripts: `extract_frames.py`, `autolabel.py`, `guess_referee.py`, `split_dataset.py`,
`calibrate.py`, `label.sh`, `patch_labelimg.py`, `run.sh`.

## Documentación (`docs/`)

| Fichero | Contenido |
|---|---|
| `docs/SPEC.md` | v1: detección + tracking + export |
| `docs/SPEC-v2.md` | v2: clasificación por equipo (color de camiseta) |
| `docs/SPEC-v3.md` | v3: calibración de pista, minimapa, heatmap |
| `docs/SPEC-v4.md` | v4: anclaje de cámara verificado por líneas |
| `docs/PLAN.md` | plan técnico de la v1 (estructura, contratos, fases) |
| `docs/TASKS.md` | histórico de tareas por fase (T1–T59) |
| `docs/NUEVO_VIDEO.md` | **checklist obligatorio al añadir un vídeo** (prefijo, extracción, etiquetado, split por grupo, prueba) |
| `README.md` | uso, flags, formatos de salida, rendimiento medido |
| `AGENTS.md` | este fichero |

Cada SPEC termina con una sección de **resultado medido** y, si procede, de criterios que fallaron.

## Convenciones de código

- Python 3.11, tipado. `ruff` (line-length 100, reglas E/F/I/UP/B/SIM/RUF) y `mypy` deben pasar.
- **Identificadores y comentarios en inglés**; mensajes de CLI/help en español.
- Configuración siempre por `Settings` + `config/default.yaml` + flags. No leer `os.environ` suelto.
- Sin estado global. Cada módulo expone funciones/clases puras o con estado explícito en un objeto.
- Tests sin red ni pesos: usar frames sintéticos (ver `tests/test_pitch.py::_pitch_like_frame`).
- No tocar `.venv/` a mano; usar `uv sync` / `uv pip install`.

## Datos y artefactos

| Ruta | Naturaleza |
|---|---|
| `videos/` | entrada del usuario. **No borrar.** |
| `dataset/images/raw/`, `dataset/labels/raw/` | **fuente de verdad del dataset.** No borrar ni regenerar sin permiso |
| `dataset/images/{train,val}`, `dataset/labels/{train,val}`, `dataset/data.yaml` | generados por `split_dataset.py` (regenerables) |
| `runs/detect/*` | modelos entrenados (`best.pt`). Caros de reproducir |
| `runs/okliga_*` | salidas del pipeline (regenerables) |
| `config/calibration_*.yaml` | calibración de pista. Difícil de rehacer (requiere clics) |
| `yolo11n.pt` | modelo por defecto del repo (regenerable por descarga) |

## Decisiones y límites conocidos

- **Clases del modelo**: `player=0, goalkeeper=1, referee=2`. `ball` se retiró por falta de instancias.
  Al usar un modelo propio hay que pasar `--detector-classes 0,1,2`, si no solo se conserva la clase 0.
- **Anclaje de cámara**: `world.py` ancla a keyframes y **verifica con las líneas amarillas** (residual
  máximo 20 px); los matches falsos de gradas/LED se rechazan. Residual mediano 15 px vs 48 px del
  encadenado. El ancla del frame de calibración nunca se descarta.
- **Calibración de pista**: el cuadrilátero actual es aproximado, así que algunas posiciones caen fuera
  de 40×20 m. Es error de calibración, no de tracking. Ver `docs/SPEC-v4.md`.
- **Deriva**: acotada por el anclaje verificado; quedan tramos de hasta ~90 px cuando no hay líneas
  visibles durante varios frames.
- **Métricas del modelo**: infladas (val sale del mismo clip). No usarlas como medida de calidad real.

## Flujo de trabajo

1. Cambios de comportamiento → `docs/SPEC*.md` aprobada antes de implementar; luego `docs/PLAN.md` y `docs/TASKS.md`.
2. Implementar con tests. `ruff`, `mypy`, `pytest` en verde antes de dar nada por terminado.
3. Actualizar `README.md` (uso) y `docs/TASKS.md` (estado) en el mismo cambio.
4. Medir, no estimar: si un cambio promete precisión, aportar el número medido y cómo se obtuvo.

## Vídeo nuevo → seguir `docs/NUEVO_VIDEO.md`

Cada vez que se copia un vídeo a `videos/`:

1. Prefijo propio y único por vídeo (`frame`, `noa`, `p2403`…). Nunca reutilizar.
2. `extract_frames.py --prefix <prefijo>` (30–80 frames).
3. `autolabel.py` → corregir clases en `label.sh`.
4. Partir train/val **por grupo/temporal**, nunca al azar (los frames contiguos son gemelos).
5. Reentrenar y **probar en un vídeo no visto**, comparando con COCO `yolo11n` como referencia.
6. Calibración de pista propia del vídeo si va a usarse el minimapa.

## Reglas duras

- No borrar `videos/`, `dataset/*/raw/`, `config/calibration_*.yaml` ni `runs/detect/*/weights/`.
- No lanzar entrenamientos largos ni descargar modelos sin confirmación.
- No reintroducir dependencias pesadas (scipy, sklearn) sin justificarlo: el proyecto es solo
  OpenCV + NumPy + Ultralytics.
- No commitear `runs/`, `.venv/`, `dataset/images/raw/` ni `*.pt`.
- No reutilizar el prefijo de extracción de otro vídeo ni mezclar dos vídeos con el mismo prefijo.
