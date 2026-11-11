# player-detect

Detección y seguimiento de jugadores en vídeos de fútbol sala con **YOLO11** + **tracking multi-objeto**.
Entrada: un vídeo. Salida: vídeo anotado (MP4 H.264) + `detections.json` + `tracks.json`.

## Instalación

Requisitos: Python 3.11/3.12 y [uv](https://docs.astral.sh/uv/). `ffmpeg` es opcional pero recomendado
(reencode a H.264; sin él el MP4 sale en `mp4v`).

```bash
uv sync
```

El modelo por defecto es `yolo11n.pt` (ya incluido en el repo). Si usas `--model yolo11m.pt` u otro,
Ultralytics lo descarga a `~/.cache/ultralytics` en la primera ejecución.

## Uso

```bash
# Clip de prueba (cámara con paneo)
uv run python -m player_detect.cli --video videos/okliga_prueba.mp4 --max-seconds 20

# Partido completo en gran angular: tiling para jugadores pequeños
uv run python -m player_detect.cli --video videos/palau.mp4 --tile 2 --imgsz 960

# Equivalente con el entry point instalado
uv run player-detect --video videos/okliga_prueba.mp4 --max-seconds 30
```

Salida en `runs/<nombre_video>/`:

```
okliga_prueba_annotated.mp4   vídeo con cajas, IDs y confianza
detections.json               detecciones por frame
tracks.json                   resumen por trayectoria
```

### Colores por equipo

Las cajas se colorean por equipo a partir del color de la camiseta:

| Etiqueta | Color | Criterio |
|---|---|---|
| `referee` | negro | cluster con menor luminancia media (V) |
| `team_a` | amarillo | el más saturado de los dos grupos restantes |
| `team_b` | azul | el otro grupo |
| `unknown` | gris | trayectoria con menos de `min_color_samples` muestras |

El muestreo usa el torso (5 %–45 % de alto, 25 %–75 % de ancho de la caja) en HSV con mediana de
píxeles, se acumula por trayectoria con una EMA y cada `classify_every` frames se agrupa con k-means
(k=3). La etiqueta es pegajosa: una vez asignada no cambia. Desactívalo con `--no-teams`.

### Flags

| Flag | Defecto | Descripción |
|---|---|---|
| `--video` | — | vídeo de entrada (obligatorio) |
| `--out` | `runs/<video>` | carpeta de salida |
| `--config` | `config/default.yaml` | YAML de parámetros |
| `--model` | `yolo11n.pt` | ruta a pesos `.pt` (propio o `yolo11n/s/m/l/x.pt`) |
| `--conf` | `0.35` | confianza mínima |
| `--iou` | `0.55` | umbral IoU de NMS |
| `--imgsz` | `960` | resolución de inferencia |
| `--batch` | `8` | frames por inferencia (se fuerza a 1 con `--tile > 1` o `--tile auto`) |
| `--tile` | `1` | rejilla NxN de inferencia para gran angular, o `auto` |
| `--tile-overlap` | `0.2` | solape relativo entre tiles |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml`, `botsort.yaml`, `ocsort.yaml` |
| `--tracker-opt` | — | opción del tracker, repetible (`KEY=VALUE`, sintaxis YAML) |
| `--device` | `auto` | `auto`, `mps`, `cuda`, `cuda:0`, `cpu` |
| `--detector-classes` | `0` | índices de clase a conservar; con un modelo propio de 4 clases usa `0,1,2` |
| `--max-seconds` | — | procesa solo los primeros N segundos |
| `--stride` | `1` | procesa 1 de cada N frames (el vídeo de salida mantiene la duración) |
| `--no-video` / `--no-json` / `--no-conf` | — | desactiva cada salida |
| `--no-teams` | — | no clasificar por equipo (color por `track_id`) |

Colores y umbrales de clasificación (BGR):

```yaml
classify_teams: true
min_box_height: 40       # cajas más bajas no aportan muestra
min_color_samples: 8     # muestras antes de etiquetar una trayectoria
classify_every: 15       # re-clasificar cada N frames
team_colors:
  team_a: [46, 190, 240]   # amarillo
  team_b: [235, 160, 60]   # azul
  referee: [20, 20, 20]    # negro
  unknown: [180, 180, 180] # gris
```

Los mismos parámetros viven en `config/default.yaml`; los flags tienen prioridad.

### Tiling automático

`--tile auto` activa `tile=2` cuando el lado mayor del vídeo supera `tile_auto_min_side`
(1920 px por defecto). Es lo que evita perder jugadores lejanos en vídeos 4K:

```bash
uv run python -m player_detect.cli --video videos/noa_penalti.mp4 --tile auto --imgsz 1280
```

### Umbrales del tracker

`--tracker-opt` inyecta cualquier clave del YAML del tracker, validando el nombre contra el
fichero real (una errata falla con la lista de claves válidas):

```bash
uv run python -m player_detect.cli --video videos/clip.mp4 \
  --conf 0.2 --tracker-opt track_high_thresh=0.15 \
  --tracker-opt track_low_thresh=0.08 --tracker-opt new_track_thresh=0.2
```

Bajar `conf` y los umbrales del tracker es lo correcto en dominios nuevos: las detecciones débiles
entran al asociador y las trayectorias se mantienen, en vez de fragmentarse en tracks de 1-2 frames.
Equivalente en `config/default.yaml`:

```yaml
tracker_params: {track_high_thresh: 0.15, track_low_thresh: 0.08, new_track_thresh: 0.2}
```

## Formatos de salida

`detections.json`

```json
{
  "video": "videos/okliga_prueba.mp4",
  "fps": 25.0, "width": 854, "height": 480, "stride": 1,
  "classes": {"player": 0, "goalkeeper": 1, "referee": 2},
  "class_counts": {"player": 320, "goalkeeper": 41, "referee": 70},
  "frames": [
    {"frame": 0, "t": 0.0,
     "detections": [{"track_id": 3, "class_id": 0, "class_name": "player",
                     "team": "team_a", "bbox": [314.0, 96.5, 388.2, 240.1],
                     "conf": 0.91, "center": [351.1, 168.3]}]}
  ]
}
```

`class_name` sale del `names` del modelo. Con un modelo propio de 3 clases distingues portero y
árbitro directamente en el JSON; con pesos COCO todas las cajas son `person` (clase 0).

`tracks.json`

```json
{
  "video": "videos/okliga_prueba.mp4",
  "track_count": 52,
  "team_counts": {"referee": 12, "team_a": 13, "team_b": 16, "unknown": 11},
  "tracks": [
    {"track_id": 3, "frames": 96, "first_t": 0.0, "last_t": 3.84,
     "path_length_px": 812.4, "mean_conf": 0.88, "mean_bbox": [330.1, 120.4, 402.0, 262.7],
     "team": "team_a", "class_name": "player", "class_counts": {"player": 95, "referee": 1},
     "mean_hsv": [57.3, 130.5, 143.2], "color_samples": 96}
  ]
}
```

`path_length_px` es el recorrido en píxeles de la imagen, no metros (no hay calibración de pista).

## Rendimiento medido (Apple M1 Pro, MPS)

| Vídeo | Ajustes | Frames | Tiempo | fps |
|---|---|---|---|---|
| `okliga_prueba.mp4` (10 s)* | `yolo11m`, `imgsz 640` | 250 | 11.3 s | 22.1 |
| `okliga_prueba.mp4` (10 s)* | `yolo11m`, `imgsz 640`, BoT-SORT | 250 | 9.4 s | 26.7 |
| `palau.mp4` (15 s, sin tiling)* | `yolo11m`, `imgsz 960` | 450 | 27.2 s | 16.6 |
| `palau.mp4` (15 s, `--tile 2`)* | `yolo11m`, `imgsz 960` | 450 | 103.2 s | 4.4 |
| `palau.mp4` (20 s, `--stride 5`)* | `yolo11m`, `imgsz 960` | 120 | 8.3 s | 14.5 |
| `okliga_prueba.mp4` (20 s, con equipos)* | `yolo11m`, `imgsz 640` | 500 | 15.2 s | 32.9 |

*medido con `yolo11m` (40 MB, ya no incluido en el repo; se descarga si lo pides con `--model`).
El tiling multiplica las detecciones encontradas (+38 % en `palau.mp4`) a cambio de ~4x de coste.
Para el partido completo de 58 min: usa `--stride`/`--max-seconds` o una GPU.

## Preparar un dataset de entrenamiento

`scripts/extract_frames.py` saca fotogramas del vídeo para etiquetar a mano y entrenar un YOLO propio:

```bash
uv run python scripts/extract_frames.py --video videos/okliga_prueba.mp4            # 1 cada 2 s
uv run python scripts/extract_frames.py --video videos/okliga_prueba.mp4 --interval 1
uv run python scripts/extract_frames.py --video videos/okliga_prueba.mp4 --start 60 --end 300
```

Guarda `dataset/images/raw/frame_000001.jpg`, `frame_000002.jpg`... a resolución original y sin
duplicados (comparación de miniaturas en gris, umbral `--dedupe-threshold 0.015`).
Solo necesita OpenCV y NumPy: `pip install opencv-python numpy`.

### Pre-etiquetar automáticamente (atajo)

```bash
uv run python scripts/autolabel.py --dry-run     # ver cuántas quedarían por hacer
uv run python scripts/autolabel.py               # escribe .txt con las personas detectadas
uv run python scripts/autolabel.py --overwrite   # regenerar (pisa lo manual)

# modelo propio (player=0, goalkeeper=1, referee=2) conservando la clase predicha
uv run python scripts/autolabel.py --model runs/detect/train-3/weights/best.pt \
  --classes 0,1,2 --keep-classes --imgsz 1280 --conf 0.30
```

Escribe todas las cajas como clase `--class-id` (def. `0` = `player`) usando la clase `person` de COCO y
respeta las etiquetas existentes. Con `--classes 0,1,2 --keep-classes` se conserva la clase que predice
el modelo propio en lugar de colapsarla a `--class-id`. Después solo hay que corregir lo que requiere
criterio: porteros y árbitros a su clase y borrar falsos positivos del público (la clase `ball` está
fuera del dataset).

### Revisar y limpiar pseudo-etiquetas

```bash
# hojas de contacto 4x4 con las cajas dibujadas (rojo player, azul portero, verde árbitro)
uv run python scripts/review_labels.py --pattern "partido_*.jpg"

# limpieza por geometría: dry-run por defecto, --write aplica con backup .txt.bak
uv run python scripts/clean_labels.py --pattern "partido_*.txt" --write
uv run python scripts/clean_labels.py --pattern "noa_*.txt" --drop-class 3 --write   # quitar ball
```

Reglas: descarta cajas de menos de 60 px o más de 600 px de alto y deja como mucho 2 porteros por frame.
Nunca mueve una caja ni cambia su clase.

### Etiquetar con labelImg

```bash
uv sync --group label          # instala labelImg (una vez)
./scripts/label.sh             # abre dataset/images/raw con las clases de dataset/classes.txt
```

En la ventana: botón **YOLO** en la barra izquierda (guarda `.txt` en `dataset/labels/raw`),
`Create RectBox` para dibujar cajas, `Ctrl+S` para guardar. Atajos: `W` crear caja, `D` siguiente imagen,
`A` anterior.

> **Al añadir un vídeo nuevo**, sigue `docs/NUEVO_VIDEO.md`: define un prefijo único por vídeo,
> extrae, pre-etiqueta, corrige clases y parte train/val por grupo (no al azar).

Dominios del dataset actual:

| Vídeo | Prefijo | Frames | Cámara | Clases |
|---|---|---|---|---|
| `okliga_prueba.mp4` | `frame` | 60 | retransmisión, paneo | player, goalkeeper, referee |
| `noa_penalti.mp4` | `noa` | 37 | 4K móvil | player, goalkeeper, referee |
| `temp.mp4` | `partido` | 63 | 4K móvil, partido completo | player, goalkeeper, referee |

Cuando tengas las etiquetas en `dataset/labels/raw/` (formato YOLO: `clase cx cy w h` normalizados),
`scripts/split_dataset.py` reparte train/val y genera `data.yaml`:

```bash
uv run python scripts/split_dataset.py --val-ratio 0.2
uv run python scripts/split_dataset.py --classes player,goalkeeper,referee,ball
uv run python scripts/split_dataset.py --check-only   # valida sin escribir nada
```

Copia (o mueve con `--move`) a `dataset/images/{train,val}` y `dataset/labels/{train,val}`, valida cada
etiqueta (5 columnas, clase en rango, coordenadas en [0,1]), omite imágenes sin etiqueta y avisa de
clases vacías. Salida: `dataset/data.yaml` listo para `model.train(data="dataset/data.yaml")`.

`--check-only` valida y muestra el mismo resumen (incluido el reparto train/val que saldría) sin
copiar ficheros ni tocar `data.yaml`, y devuelve código 1 si hay etiquetas inválidas o alguna clase
por debajo de `--min-instances` (50 cajas por defecto, 0 lo desactiva). Úsalo antes de etiquetar
para saber cuántas imágenes faltan:

```bash
uv run python scripts/split_dataset.py --check-only --min-instances 30
```

Si no pasas `--classes`, se toman los nombres de `dataset/data.yaml` cuando ya existe.

## Usar un modelo entrenado por ti

```bash
uv run python -m player_detect.cli --video videos/okliga_prueba.mp4 \
  --model runs/detect/train-3/weights/best.pt --detector-classes 0,1,2
```

`--detector-classes` es imprescindible: el detector filtra por índice de clase y, por defecto, solo
conserva la 0 (jugador). Con clases `player=0, goalkeeper=1, referee=2, ball=3` hay que pasar `0,1,2`
(el balón no debe entrar en el minimapa ni en la clasificación por equipos).

## Minimapa top-down

Con una calibración de pista el pipeline proyecta a cada jugador sobre un plano cenital:

```bash
# 1) Calibrar una vez: 4 clics en esquinas distintas del suelo (el orden da igual)
#    Z deshace el último punto, R reinicia, ENTER guarda, ESC cancela.
uv run python scripts/calibrate.py --video videos/okliga_prueba.mp4 --width 40 --height 20

# 2) Procesar con el minimapa
uv run python -m player_detect.cli --video videos/okliga_prueba.mp4 --max-seconds 30 \
  --calibration config/calibration_okliga_prueba.yaml
```

O en una sola orden, que calibra solo si hace falta y luego procesa:

```bash
./scripts/calibrate_and_detect.sh videos/noa_penalti.mp4 30        # 30 s, calibra si falta el YAML
./scripts/calibrate_and_detect.sh videos/noa_penalti.mp4 30 --force --open
./scripts/calibrate_and_detect.sh videos/noa_penalti.mp4 30 \
  --points "1500,1400 3000,1350 3200,2000 1300,2100"    # sin GUI
```

El script reutiliza `config/calibration_<video>.yaml` si existe y es de ese vídeo con `frame: 0`; si no,
calibra (GUI o `--points`), lanza el pipeline con `--tile auto` y el modelo de 3 clases, y al final lista
los artefactos y avisa si faltan `minimap.mp4` o `heatmap.png`. Opciones: `--force`, `--points`,
`--calibration FILE` (varios clips de la misma cámara) y `--open`.

Genera además de lo habitual: minimapa incrustado en el vídeo, `minimap.mp4` (800x400 cenital),
`heatmap.png` (un panel por equipo), `heatmap_counts.json` y `positions.json` con `x_m`/`y_m` por frame.

Los clics repetidos (doble clic) y los que caen a menos de 8 px de otro se descartan solos, así que la
calibración ya no falla con `degenerate calibration: duplicated points`. Si no puedes usar la GUI, pasa
los píxeles a mano: `--points "x1,y1 x2,y2 x3,y3 x4,y4"`.
Flags: `--no-minimap`, `--no-heatmap`, `--minimap-size 0.22`.

El orden de los clics no importa: `config` se normaliza a TL/TR/BR/BL al cargar. Los jugadores fuera de
la pista se dibujan en el anillo de margen (2 m), no desaparecen; solo se descartan proyecciones a más de
`pitch_max_offset_m` (15 m). El resumen indica cuántos puntos van al anillo y cuántos se descartan.

> **Precisión**: la cámara se ancla a las **líneas amarillas de la pista** cada 5 frames y cada anclaje se
> verifica contra las líneas del frame de calibración (`world_anchor`, residual máximo 20 px). Medido en
> 600 frames: residual mediano 15 px frente a 48 px del encadenado simple, y los matches falsos de gradas/LED
> se rechazan. El error restante lo domina la calidad de la calibración de pista, no la deriva.
> Ver `docs/SPEC-v4.md`.
> `config/calibration_okliga_prueba.demo.yaml` es una calibración aproximada solo para pruebas.

## Limitaciones

- Cámara con paneo (`okliga_prueba.mp4`): sin compensación de movimiento global hay fragmentación de IDs
  (~38 trayectorias en 10 s para ~10 jugadores simultáneos). `--tracker botsort.yaml` no mejora de forma
  significativa en este clip.
- Gran angular (`palau.mp4`): sin `--tile 2/3` los jugadores lejanos (~15 px) se pierden.
- La clasificación por equipo depende del color de camiseta: un equipo de camiseta muy oscura
  (azul marino) puede confundirse con el árbitro negro. Los primeros frames de cada trayectoria
  salen en gris hasta acumular `min_color_samples`.
- Solo clase `person`: no lee dorsales.
- Minimapa: deriva acumulada del homography de cámara (ver sección Minimapa top-down).
- Sin homografía: no hay distancia en metros, velocidad real ni heatmap.

## Desarrollo

```bash
uv run pytest        # 50 tests, sin red ni pesos
uv run ruff check src/ tests/
uv run mypy src/
```

Estructura: `src/player_detect/{config,video,detector,tracking,teams,camera,pitch,minimap,draw,export,pipeline,cli}.py`.
Diseño completo en `docs/`: `SPEC.md`, `SPEC-v2.md`, `SPEC-v3.md`, `SPEC-v4.md`, `PLAN.md` y `TASKS.md`.
