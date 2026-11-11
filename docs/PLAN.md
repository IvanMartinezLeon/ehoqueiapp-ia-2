# PLAN — Implementación de detección y seguimiento de jugadores

Referencia: `docs/SPEC.md`. Stack: Python 3.11 + `uv` + Ultralytics YOLO11 + OpenCV.

## 1. Estructura del proyecto

```text
ehoqueiapp-ia-2/
├── SPEC.md
├── PLAN.md
├── TASKS.md
├── README.md
├── pyproject.toml                # uv, deps, ruff/mypy/pytest
├── config/
│   └── default.yaml              # parámetros por defecto
├── videos/                       # entradas del usuario (ya existe)
├── runs/                         # salidas (gitignored)
├── src/player_detect/
│   ├── __init__.py
│   ├── config.py                 # dataclass Settings + carga YAML + merge de flags
│   ├── video.py                  # VideoSource (lectura streaming) + VideoSink (escritura robusta)
│   ├── detector.py               # YoloDetector: predict por lotes, clase person, tiling opcional
│   ├── tracking.py               # tracker ByteTrack/BoT-SORT vía ultralytics.track
│   ├── pipeline.py               # orquestación: frames → detección → tracking → render → export
│   ├── draw.py                   # renderizado de cajas/IDs (paleta determinista por ID)
│   ├── export.py                 # detections.json + tracks.json
│   └── cli.py                    # argparse + logging + resumen final
├── scripts/
│   └── run.sh                    # atajos para los dos vídeos de ejemplo
└── tests/
    ├── test_export.py            # estructura JSON y agregación de trayectorias
    ├── test_draw.py              # paleta determinista, no muta el frame de entrada
    └── test_config.py            # merge flags > yaml > defaults
```

## 2. Flujo de datos

```text
VideoSource ──frame(batch de N)──▶ YoloDetector.predict
                                        │
                              (opcional tiling + NMS fusión)
                                        ▼
                              tracking (ByteTrack, persistente)
                                        ▼
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
             VideoSink.write(annotated)          ExportCollector.add(frame)
                     │                                     │
                out.mp4                          detections.json / tracks.json
```

`pipeline.run(settings)` es una función pura respecto a la I/O: recibe un `Settings`,
devuelve un `RunSummary` (frames, tiempo, ids, fps medio). La CLI imprime el resumen.

## 3. Decisiones técnicas

| Decisión | Elección | Motivo |
|---|---|---|
| Modelo | `yolo11m.pt` por defecto, `yolo11s`/`yolo11x` por flag | Equilibrio precisión/velocidad en M1 Pro |
| Clase filtrada | `classes=[0]` (person COCO) | Evita balones, porterías, etc. |
| Tracking | `model.track(..., persist=True, tracker="bytetrack.yaml")` | API nativa de Ultralytics, sin dependencias extra |
| Batches | `--batch 8` acumula frames | Mejor uso de MPS/GPU |
| Tiling | `--tile N`: rejilla con 20 % solape, `torchvision.ops.nms` para fusionar | Necesario para `palau.mp4` |
| Escritura de vídeo | `cv2.VideoWriter` (mp4v) → reencode con ffmpeg a H.264 | Compatibilidad QuickTime |
| Progreso | `tqdm` a stderr | No ensucia stdout (reservado para resumen) |
| Config | YAML + dataclass frozen `Settings` con `replace()` | Tipado, sin mutación |
| Determinismo de color | `track_id * 37 % 180` en HSV | Mismo ID → mismo color entre ejecuciones |

## 4. Contrato de módulos

```python
# config.py
@dataclass(frozen=True)
class Settings:
    video: Path; out_dir: Path
    model: str = "yolo11m.pt"; conf: float = 0.35; iou: float = 0.5
    imgsz: int = 960; batch: int = 8; tile: int = 1
    tracker: str = "bytetrack.yaml"; device: str = "auto"
    max_seconds: float | None = None; stride: int = 1

def load_settings(argv: list[str]) -> Settings: ...
```

```python
# detector.py
class YoloDetector:
    def __init__(self, settings: Settings) -> None: ...
    def track(self, frames: list[np.ndarray]) -> list[list[Detection]]: ...
```

```python
# export.py
@dataclass
class Detection:  # frame, t, track_id, bbox, conf
    ...

class ExportCollector:
    def add(self, frame_idx: int, t: float, dets: list[Detection]) -> None: ...
    def write(self, detections_path: Path, tracks_path: Path) -> TrackSummary: ...
```

## 5. Fases

1. **Bootstrap** — `pyproject.toml`, `uv sync`, esqueleto de módulos, `ruff`/`mypy`/`pytest` configurados.
2. **Pipeline mínimo** — lectura, detección sin tracking, render simple, escritura MP4. Validar en 10 s de `okliga_prueba.mp4`.
3. **Tracking** — ByteTrack, IDs en render, `detections.json` + `tracks.json`.
4. **Tiling + stride + batching** — modo gran angular para `palau.mp4`.
5. **Robustez y cierre** — fallback ffmpeg, tests, README con ejemplos y tabla de rendimiento medida.

## 6. Estrategia de verificación

- Tests unitarios sin red ni pesos: export, draw, config (usan detecciones sintéticas).
- Test de integración marcado `@pytest.mark.slow` y `skipif` si no hay pesos en caché.
- Verificación manual: los criterios de aceptación de `docs/SPEC.md` §6 ejecutados sobre los dos vídeos de `videos/`.

## 7. Riesgos de implementación

- **`model.track` + batches**: Ultralytics soporta tracking con batch, pero los IDs son por stream;
  si da problemas, se procesa frame a frame en modo tracking y el batch se usa solo en detección pura.
- **MPS y ByteTrack**: el tracker corre en CPU (NumPy); el cuello de botella es la inferencia.
- **Memoria en `--tile 3`**: 9 crops por frame a 1280 px → vigilar con `--batch 1`.
