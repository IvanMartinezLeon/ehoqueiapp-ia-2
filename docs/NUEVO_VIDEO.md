# Checklist: por cada vídeo nuevo que se copia a `videos/`

Procedimiento obligatorio cada vez que se añade un vídeo. Sin esto el dataset se desordena y las
métricas dejan de significar algo.

## 0. Regla de oro: un prefijo por vídeo

Cada vídeo tiene su **prefijo de extracción**, derivado de su nombre y único:

| Vídeo en `videos/` | Prefijo | Ficheros generados |
|---|---|---|
| `okliga_prueba.mp4` | `frame` | `dataset/images/raw/frame_000001.jpg` … |
| `noa_penalti.mp4` | `noa` | `dataset/images/raw/noa_000001.jpg` … |
| `partido_2024_03.mp4` | `p2403` | `p2403_000001.jpg` … |

- El prefijo **nunca** se reutiliza entre vídeos: es lo que permite separar dominios al partir train/val.
- No mezclar dos vídeos con el mismo prefijo: los `frame_000001` se pisarían.
- El contador reinicia en cada ejecución, así que **cada vídeo tiene su propio prefijo**.

## 1. Situar el vídeo y ver qué es

```bash
cp ~/Descargas/mi_partido.mp4 videos/
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate \
  -of default=noprint_wrappers=1 videos/mi_partido.mp4
```

Anotar resolución, fps y duración: define cuántos frames saldrán y si el dominio es nuevo
(cámara, iluminación, cancha, equipaciones).

## 2. Extraer frames con prefijo propio

```bash
# ~0.5–1 s entre frames para retransmisiones largas; 0.2 s en clips cortos
uv run python scripts/extract_frames.py --video videos/mi_partido.mp4 \
  --interval 1 --prefix p2403 --start 0 --end 300
```

Objetivo: 30–80 frames por vídeo. Más frames del mismo plano no aportan diversidad.

## 3. Pre-etiquetar (no etiquetar a mano desde cero)

```bash
uv run python scripts/autolabel.py --conf 0.30 --imgsz 1280
```

Usa `yolo11n.pt` (COCO). Escribe todas las cajas como clase `0` (`player`) y **respeta** las
etiquetas existentes. Salta las imágenes que ya tienen `.txt`.

Si ya hay un modelo del proyecto, conserva sus clases nativas (portero/árbitro):

```bash
uv run python scripts/autolabel.py --model runs/detect/train-3/weights/best.pt \
  --classes 0,1,2 --keep-classes --imgsz 1280 --conf 0.30
```

> Si el vídeo es de un dominio muy distinto y el detector falla, subir `--imgsz` o usar
> `--model runs/detect/train-3/weights/best.pt` si ya tienes un modelo del mismo dominio.

## 4. Revisar y corregir clases

```bash
./scripts/label.sh
```

- `goalkeeper` (1) al portero · `referee` (2) al árbitro · borrar cajas de público/banquillo.
- Añadir el balón solo si vas a mantener esa clase (hoy está fuera del dataset).

## 5. Revisar el dataset completo antes de partir

```bash
uv run python -c "
import sys, collections; from pathlib import Path; sys.path.insert(0,'scripts')
from split_dataset import validate_label_file
classes=['player','goalkeeper','referee']
for f in sorted(Path('dataset/labels/raw').glob('*.txt')):
    ok,boxes,_=validate_label_file(f,3)
    print(f.name, boxes if ok else 'INVALIDA')
" | tail -20
```

Comprobar que no hay imágenes sin etiqueta y que el reparto por clase sigue teniendo sentido.

## 6. Partir train/val **por grupo/temporal**, no al azar

Frames a 1 s de distancia son casi idénticos: un reparto aleatorio pone gemelos en train y val e
infla la mAP (fue la causa de que `train-2` pareciera bueno y fallara en `noa_penalti.mp4`).

```bash
uv run python scripts/split_dataset.py --group-by prefix --val-ratio 0.2 --clean
```

`--group-by prefix` es el valor por defecto: agrupa por prefijo (un vídeo = un grupo) y manda a val el
20 % final de cada clip, así los frames gemelos nunca caen en lados distintos.

## 7. Reentrenar y **probar en un vídeo no visto**

```bash
uv run yolo detect train data=dataset/data.yaml model=yolo11n.pt \
  epochs=60 imgsz=960 batch=8 patience=20 mosaic=0.5

uv run python -m player_detect.cli --video videos/mi_partido.mp4 \
  --model runs/detect/train-N/weights/best.pt --detector-classes 0,1,2 --no-video
```

Referencia de la prueba anterior (dominio nuevo = 4K móvil, modelo solo con retransmisión):

| Modelo | Detecciones | Frames sin detectar |
|---|---|---|
| `train-2` (solo retransmisión) | 73 / 229 | 166 (73 %) |
| COCO `yolo11n` | 653 / 229 | 0 |

Si el modelo propio sigue muy por debajo de COCO en un vídeo nuevo, **falta ese dominio en el
dataset**, no más epochs.

## 8. Actualizar el registro

- Añadir línea a la tabla de dominios del `README.md` (vídeo → prefijo → frames → clases).
- Anotar en `docs/TASKS.md` si el vídeo revela un fallo nuevo.
- Si el vídeo servirá para el minimapa, crear su calibración propia:
  `uv run python scripts/calibrate.py --video videos/mi_partido.mp4 --width 40 --height 20`
  (una calibración **por vídeo**, no se reutiliza entre cámaras).

## Errores que ya hemos cometido

| Error | Consecuencia |
|---|---|
| Repartir train/val al azar con frames casi duplicados | mAP 0.971 que no se sostiene en vídeo nuevo |
| Entrenar con una única retransmisión | el modelo falla en 4K desde móvil (73 vs 653 detecciones) |
| Mezclar vídeos con el mismo prefijo | se pisan ficheros y se pierden etiquetas |
| Usar la calibración de un vídeo en otro | posiciones del minimapa sin sentido |
| Dar por buena la mAP sin probar en vídeo no visto | falsa sensación de calidad |
