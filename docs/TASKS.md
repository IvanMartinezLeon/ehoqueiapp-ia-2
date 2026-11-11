# TASKS — Detección y seguimiento de jugadores

Leyenda: `[ ]` pendiente · `[~]` en curso · `[x]` hecho

**Estado: v1 completada.** Desviaciones respecto al plan:
- `BYTETracker` de ultralytics 8.4 ya no acepta `frame_rate`; la fábrica de trackers solo recibe `args`.
- Los trackers se instancian a mano (`YAML.load` + `IterableSimpleNamespace`) para poder unificar detección con tiles y tracking en un único flujo.
- `--run` -> la salida por defecto es `runs/<stem>/` tal y como se especificó.

## Fase 1 — Bootstrap

- [x] T1. Crear `pyproject.toml` (Python 3.11, deps: ultralytics, opencv-python, numpy, pyyaml, tqdm; dev: pytest, ruff, mypy) con `uv`.
- [x] T2. `uv sync` y verificar import de `ultralytics` + `torch` en MPS.
- [x] T3. Crear esqueleto de `src/player_detect/` con módulos vacíos tipados.
- [x] T4. `config/default.yaml` con parámetros por defecto.

## Fase 2 — Pipeline mínimo

- [x] T5. `video.py`: `VideoSource` (cv2 streaming, fps, size, total_frames) y `VideoSink` (mp4v + reencode ffmpeg).
- [x] T6. `config.py`: `Settings` frozen + carga YAML + merge con flags (`--flag` gana).
- [x] T7. `detector.py`: `YoloDetector` con filtro `classes=[0]`, selección de device `auto`.
- [x] T8. `draw.py`: cajas + etiqueta con ID/conf, paleta determinista.
- [x] T9. `pipeline.py` v0: frames → detect → draw → write (sin tracking, sin export).
- [x] T10. `cli.py` con `--video`, `--out`, `--max-seconds`, resumen final a stdout.
- [x] T11. Validar 10 s de `videos/okliga_prueba.mp4` y comprobar reproducción del MP4.

## Fase 3 — Tracking y export

- [x] T12. `tracking.py`: integrar `model.track(..., persist=True, tracker=...)`; fallback a detección frame a frame si el batch falla.
- [x] T13. `export.py`: `ExportCollector` → `detections.json` (por frame) y `tracks.json` (por ID: frames, t0/t1, longitud de trayectoria, bbox media).
- [x] T14. Tests `test_export.py` con detecciones sintéticas (estructura, agregación, IDs no nulos).
- [x] T15. Verificar criterio de aceptación 3 sobre 20 s de `okliga_prueba.mp4`.

## Fase 4 — Modo gran angular

- [x] T16. `detector.py`: tiling NxN con 20 % solape + fusión NMS (`--tile`).
- [x] T17. `--stride` (procesar 1 de cada N frames, manteniendo fps de salida).
- [x] T18. Batching real (`--batch`) y ajuste automático a 1 cuando `tile > 1`.
- [x] T19. Validar criterio de aceptación 5 en 10 s de `videos/palau.mp4`.

## Fase 5 — Cierre

- [x] T20. `scripts/run.sh` con comandos para ambos vídeos.
- [x] T21. `.gitignore` (`runs/`, `.venv/`, `*.pt`, `__pycache__`).
- [x] T22. `README.md`: instalación, uso, flags, ejemplos y tabla de rendimiento medida en M1 Pro.
- [x] T23. `ruff check` + `mypy` + `pytest` en verde.
- [x] T24. Ejecutar los 6 criterios de aceptación del `docs/SPEC.md` §6 y anotar resultados.

## Fase 6 — v2: clasificación por equipo (`docs/SPEC-v2.md`)

- [x] T25. `teams.py`: muestreo del torso en HSV (mediana, hue como vector unitario) por caja.
- [x] T26. `teams.py`: EMA por `track_id` y k-means determinista (k=3, `n_init=3`, semilla fija).
- [x] T27. `teams.py`: etiquetado `referee` = menor V, `team_a`/`team_b` por saturación, etiquetas pegajosas.
- [x] T28. `draw.annotate` acepta `colors` por `track_id`; texto blanco sobre cajas oscuras.
- [x] T29. `config`/`config/default.yaml`: `classify_teams`, `min_color_samples`, `classify_every`, `min_box_height`, `team_colors`.
- [x] T30. `export.py`: `team` por detección y `team`/`mean_hsv`/`color_samples`/`team_counts` en `tracks.json`.
  (Corregido bug: la etiqueta se resolvía al añadir el frame, antes de clasificar.)
- [x] T31. `cli.py`: flag `--no-teams` y recuento por equipo en el resumen.
- [x] T32. `tests/test_teams.py`: 15 tests (muestreo, k-means determinista, árbitro = más oscuro, sticky).
- [x] T33. Verificación visual: amarillo / azul / negro en `okliga_prueba.mp4`; `--no-teams` reproduce v1.

## Fase 7 — v3: minimapa top-down (`docs/SPEC-v3.md`)

- [x] T34. `camera.py`: `CameraTracker` (LK + RANSAC), `compose`, `to_reference`.
- [x] T35. `pitch.py`: `Calibration` (YAML ida/vuelta), `RinkProjector`, `foot_point`.
- [x] T36. `minimap.py`: `rink_canvas`, `build_minimap`, `overlay`, `HeatmapAccumulator`.
- [x] T37. `scripts/calibrate.py`: clics en 4 puntos (o `--points`) → `config/calibration_<stem>.yaml`.
- [x] T38. `pipeline.py`: `_RinkState`, minimapa incrustado, `minimap.mp4`, heatmap, `positions.json`.
- [x] T39. `export.py`: `rink_m` por detección, `rink_path_m`/`mean_position_m` por trayectoria.
- [x] T40. `cli.py`: `--calibration`, `--no-minimap`, `--no-heatmap`, `--minimap-size`.
- [x] T41. `tests/test_pitch.py`: 16 tests (homografía sintética, proyección, minimapa, heatmap).
- [x] T42. Bugs corregidos: heatmap acumulaba por `track_id` en vez de por equipo; `positions_payload` leía `team` antes de resolverlo; `overlay` rompía con frames diminutos.
- [x] T43. Deriva medida y documentada (criterio 5 **falla**: 48 px a 100 frames).
- [x] T44. ~~**v4**: anclaje por líneas de pista~~ → hecho en Fase 9 (`docs/SPEC-v4.md`).

## Fase 8 — Correcciones tras primera calibración real

- [x] T45. `pitch.order_quad`: orden TL/TR/BR/BL independiente del orden de clics; rechazo de cuadriláteros cruzados.
- [x] T46. `load_calibration` normaliza calibraciones ya guardadas con clics cruzados (las de v3 rompían la homografía: puntos a ±1900 m).
- [x] T47. Proyección: ya no se descarta lo que cae fuera de la pista; se dibuja en el anillo de margen y solo se descarta más allá de `pitch_max_offset_m` (15 m). Cobertura 51 % → 85 %.
- [x] T48. Anillo de margen de 2 m en el minimapa y en `minimap.mp4`; puntos fuera se recortan al borde del lienzo.
- [x] T49. `VideoSink.write` valida el tamaño del frame (antes escribía 0 frames en silencio y `minimap.mp4` salía corrupto con 257 bytes).
- [x] T50. Heatmap: 4 paneles fijos (team_a, team_b, referee, unknown), cada uno pintado en el color de su equipo, con total de puntos; `heatmap_counts.json` incluye `totals`.
- [x] T51. Resumen con diagnóstico de proyección (en anillo / descartados / sin referencia) y aviso si la cobertura < 60 %.
- [x] T52. ~~**v4 (crítico)**: anclaje por líneas~~ → hecho en Fase 9: residual mediano 15 px (antes 48 px).

## Fase 9 — v4: anclaje verificado por líneas (`docs/SPEC-v4.md`)

- [x] T53. `world.py`: `floor_line_mask` (filtro por forma para ignorar camisetas amarillas).
- [x] T54. `world.py`: `line_residual` como función de verificación.
- [x] T55. `world.py`: `WorldTracker` con ancla fija + keyframes recientes + verificación + arrastre LK.
- [x] T56. Integración en `pipeline._RinkState` (`world_anchor`, `world_max_residual_px`, `world_anchor_interval`).
- [x] T57. 7 tests nuevos (máscara, residual, ancla fija, stats).
- [x] T58. Medición: residual mediano 15.2 px vs 48 px del encadenado; 89 matches falsos rechazados.
- [ ] T59. **PENDIENTE**: recalibrar la pista con líneas reales para que las posiciones caigan dentro de 40×20 m.
- [ ] T60. **PENDIENTE**: ampliar el dataset (3–5 partidos, otras canchas); hoy son 60 imágenes de un solo clip.
- [ ] T61. **PENDIENTE**: clase `ball` (se retiró por falta de instancias).
- [ ] T62. **PENDIENTE**: inicializar repositorio git y primer commit (el proyecto no está versionado).
- [x] T63. `split_dataset.py --group-by prefix` (split por grupo/bloque temporal). Hoy el reparto es aleatorio y mete frames gemelos en train y val, inflando la mAP.
- [x] T64. Documentado el procedimiento por vídeo en `docs/NUEVO_VIDEO.md` + regla de prefijo único.
- [x] T65. `autolabel.py --classes` + `--keep-classes`: con modelo propio escribe la clase predicha (player/portero/árbitro) en vez de colapsar todo a `--class-id`. Tests en `tests/test_autolabel.py`.
- [ ] T66. **PENDIENTE**: extraídos 63 frames de `videos/temp.mp4` (prefijo `partido`, 4K@60, cada 10 s con dedupe). Falta pre-etiquetar, corregir clases y reentrenar.
- [x] T67. `scripts/review_labels.py`: hojas de contacto 4×4 con cajas por clase. 63 frames de `partido_*` → 4 hojas en `runs/label_review/`.
- [x] T68. `scripts/clean_labels.py`: limpieza por geometría (h<60 px, h>600 px, máx. 2 porteros/frame) con dry-run por defecto y backup `.txt.bak`. Medido en `partido_*`: 931 → 873 (tras edición manual) → **750** cajas; 123 descartadas.
- [x] T69. Clase `ball` eliminada del dataset: 13 cajas de `noa_*.txt` borradas con `clean_labels.py --drop-class 3` (backup `.bak`). `data.yaml` ahora `nc: 3`.
- [x] T70. Split por grupo: `frame 48/12, noa 30/7, partido 50/13` → 128 train / 32 val, 1356 cajas (`player=1024, goalkeeper=143, referee=189`), 0 etiquetas inválidas.
- [x] T71. Reentrenado `train-3` (`yolo11n`, 58 epochs, early stop; mejor epoch 38): val P 0.811 / R 0.700 / mAP50 0.813 / mAP50-95 0.568. Pesos en `runs/detect/train-3/weights/best.pt`.
- [x] T72. Prueba en la cola (val) de `partido_*` con GT: `train-3` recall 0.52 / precision 0.69 vs `train-2` 0.42 / 0.92 y COCO `yolo11n` 0.26 / 0.33. `train-3` recupera porteros (6 cajas donde `train-2` daba 1).
- [x] T73. Resumen del CLI independiente del export: `detecciones`, `trayectorias`, `por clase` y `equipos` ahora se acumulan en `pipeline.run` en vez de leerse del `ExportCollector`. Antes, con `--no-json` el resumen decía `detecciones: 0` / `trayectorias: 0` y omitía los tracks sin clasificar en `equipos` (`unknown`). Verificado: 3 s de `temp_demo_720p` dan 339/19/`player=334, referee=5`/`unknown=11` con y sin `--no-json`.
- [x] T75. `scripts/calibrate.py` ya no es frágil: `_accepts_click` descarta clics a menos de 8 px de otro (mata el doble clic que producía `degenerate calibration: duplicated points`), `Z`/`Backspace` deshace el último punto, `R` reinicia y `ENTER` confirma los 4. La ventana redibuja marcadores, polígono y contador. 5 tests nuevos en `tests/test_calibrate.py`.
- [x] T76. **Bug**: `pitch.order_quad` fallaba con `degenerate calibration: duplicated points` en pistas en perspectiva. La regla clásica de extremos `x+y` / `x-y` colapsaba TL/BR y TR/BL sobre la misma esquina cuando el lado cercano es mucho más ancho que el lejano (caso normal al filmar desde un fondo). Ahora ordena por ángulo alrededor del centroide, ancla la primera esquina al mínimo `x+y` y fija la orientación con el área firmada. Añadidos `_signed_area` y 2 tests (trapecio en perspectiva con 4 órdenes de clic; punto interior rechazado).
