# SPEC v3 — Minimapa top-down con coordenadas de pista

Estado: **implementada** (v3). Amplía `docs/SPEC.md` (§2 "Fuera de alcance v1").

## 1. Objetivo

Proyectar a cada jugador sobre un plano cenital de la pista y mostrarlo como **minimapa** con un punto
por jugador coloreado por equipo, además de un vídeo cenital independiente y un heatmap de ocupación.

Entrada nueva: un fichero de **calibración** (4 puntos imagen ↔ 4 puntos de pista en metros), generado una
vez por vídeo con un script interactivo.

## 2. Alcance

### Incluido (v3)

- `scripts/calibrate.py`: clic sobre 4 puntos de un frame + coordenadas reales → `config/calibration_<stem>.yaml`.
- Compensación de movimiento de cámara por frame (LK + `findHomography` RANSAC) para que la proyección
  siga siendo válida con cámara en paneo.
- Posición de cada jugador = **punto de los pies** (centro-x, borde inferior de la caja) → metros de pista.
- Minimapa incrustado en el vídeo anotado (esquina inferior derecha) + `minimap.mp4` cenital limpio.
- `heatmap.png` con densidad de ocupación por equipo.
- Coordenadas de pista en la exportación JSON.
- Degradación limpia: sin calibración el pipeline se comporta como la v2.

### Fuera de alcance (v3)

- Detección automática de líneas de pista / calibración sin clics.
- Re-anclaje automático contra deriva del homography (se documenta y se mide).
- Distancia recorrida en metros por jugador, velocidad, posesión, análisis táctico.
- Cámara que se mueva físicamente (grúas, zoom brusco de realización) sin plano estable.

## 3. Requisitos funcionales

- **RF-1** `scripts/calibrate.py --video V [--frame N] [--width W --height H]` abre el frame, deja clicar 4
  puntos y escribe YAML con `image_points` (píxeles) y `rink_points` (metros, en orden coherente).
  Alternativa sin GUI: `--points "x1,y1 x2,y2 x3,y3 x4,y4"`.
- **RF-2** El YAML se valida: 4 puntos, sin colinealidad degenerada (`findHomography` con 4 puntos no nulo).
- **RF-3** `CameraTracker.update(prev_gray, gray) -> np.ndarray | None` devuelve la homografía imagen→imagen
  entre frames. Si los inliers < `min_inliers` (def. 40) devuelve `None` y se reutiliza la última válida.
- **RF-4** La homografía acumulada desde el frame de calibración compone todas las transformaciones
  intermedias en orden, sin reordenar frames ni perder estado entre batches.
- **RF-5** `RinkProjector.to_rink(pixel_xy, frame_index) -> (x_m, y_m) | None` devuelve `None` si la
  proyección cae fuera de la pista (`margin` configurable) o si el denominador homogéneo es ~0.
- **RF-6** En el vídeo anotado se incrusta un minimapa de ancho = `size_ratio` (def. 0.22) × ancho de frame,
  con la pista, la línea central, el círculo central y un punto por jugador con el color de su equipo.
- **RF-7** `minimap.mp4` reproduce el minimapa a 800x400 con los mismos fps y duración.
- **RF-8** `heatmap.png` acumula una rejilla `heatmap_grid` (def. 80x40 celdas) por equipo y la pinta
  con `applyColorMap`; se guarda también `heatmap_counts.json` con la matriz cruda.
- **RF-9** `tracks.json` añade por trayectoria `rink_path_m` (recorrido en metros) y
  `mean_position_m`; `positions.json` añade `(frame, track_id, x_m, y_m)` por frame.
- **RF-10** Flags: `--calibration PATH` (activa minimapa/heatmap), `--no-minimap`, `--heatmap/--no-heatmap`,
  `--minimap-size FLOAT`. Sin `--calibration` no se escribe ninguna salida nueva.

## 4. Requisitos no funcionales

- Coste añadido < 10 % del tiempo total (LK sobre 300 puntos + dibujo 2D).
- Sin dependencias nuevas (OpenCV + NumPy).
- Determinista: mismo vídeo + misma calibración → mismos puntos (RANSAC con semilla fija).

## 5. Criterios de aceptación

1. `python scripts/calibrate.py --video videos/okliga_prueba.mp4 --width 40 --height 20` genera un YAML que
   el pipeline carga sin error y reproyecta sus 4 puntos imagen↔pista con error < 3 px.
2. Con `--calibration`, el vídeo anotado muestra el minimapa y los puntos se mueven de forma coherente con
   el vídeo (comprobación visual en ≥ 3 frames separados).
3. `minimap.mp4` existe, dura lo mismo que el tramo procesado y no está vacío.
4. `heatmap.png` no es una imagen uniforme (desviación típica de píxeles > 5).
5. **Deriva**: comparando la homografía encadenada del frame 0 al 250 contra la estimada directamente 0→250,
   el error medio de reproyección de 100 puntos del frame 0 es < 15 px. Si no se cumple, se documenta el
   límite y se reduce el alcance (minimapa solo para tramos cortos).
6. Repetir la ejecución produce el mismo `positions.json` (determinismo).
7. `ruff`, `mypy` y `pytest` en verde; los tests nuevos cubren composición de homografías, proyección
   ida/vuelta, rechazo de puntos fuera de pista y acumulación del heatmap.

## 6. Limitaciones y riesgos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Deriva del homography encadenado en clips largos | Puntos desplazados en los últimos frames | Criterio de aceptación 5 lo mide; `--max-seconds` para tramos |
| Solo 4 puntos de calibración en una vista parcial | Error de escala si los puntos no son coplanares a la pista | El script valida la forma; se documentan buenas prácticas (usar líneas del suelo) |
| Oclusión total (plano del público o repetición) | LK pierde el plano | `min_inliers` alto → se mantiene la última homografía válida |
| Jugadores fuera del encuadre con el paneo | Sin posición | Se omiten del minimapa ese frame (documentado) |

## 7. Resultado de la verificación (implementado)

| Criterio | Resultado |
|---|---|
| 1. YAML de calibración válido y reproyección < 3 px | ✅ error 0.000 m en los puntos de calibración |
| 2. Minimapa coherente en ≥ 3 frames | ✅ incrustado en la esquina inferior derecha, puntos por equipo |
| 3. `minimap.mp4` con la misma duración | ✅ 800x400, 25 fps, 300 frames = 12 s |
| 4. `heatmap.png` no uniforme | ✅ 3 paneles (unknown/team_a/team_b), std = 43.2 |
| 5. **Deriva < 15 px a 250 frames** | ❌ **FALLA** (ver abajo) |
| 6. Determinismo de `positions.json` | ✅ misma calibración → mismas posiciones |
| 7. ruff + mypy + pytest | ✅ 50 tests |

### Deriva medida (okliga_prueba.mp4, 25 fps)

| Horizonte | Inliers de la referencia directa (ORB) | Error medio | Error p95 |
|---|---|---|---|
| 10 frames (0.4 s) | 2710 | 3.9 px | 10.0 px |
| 25 frames (1.0 s) | 1632 | 10.1 px | 23.2 px |
| 50 frames (2.0 s) | 752 | 19.3 px | 40.8 px |
| 75 frames (3.0 s) | 483 | 52.1 px | 131.5 px |
| 100 frames (4.0 s) | 627 | 48.2 px | 125.4 px |

La composición de homografías es correcta (test sintético con traslación conocida: error 0.07 px en 40 frames),
así que la deriva viene del ruido acumulado del seguimiento de features, no de un fallo de implementación.

**Actualización (v4)**: el anclaje verificado por líneas (`docs/SPEC-v4.md`) baja el residual mediano de 48 px a 15 px.

**Consecuencia original (v3)**: el minimapa era fiable en tramos de ~1 s y se degrada a partir de ~2 s. No es apto para
posicionamiento absoluto en clips largos. La solución correcta es anclar cada frame a las **líneas de la pista**
(mask amarilla + ajuste del modelo de rink), que elimina la deriva por construcción. Queda como v4.
