# SPEC v2 — Clasificación por equipo (color de camiseta)

Estado: **aprobada e implementada** (v2.0). Amplía `docs/SPEC.md` (§2 "Fuera de alcance v1").

## 1. Objetivo

Cada caja se dibuja con el color de su equipo, de forma que en un vídeo haya **tres colores de caja**:

| Etiqueta | Color por defecto | Criterio |
|---|---|---|
| `referee` | negro `(20, 20, 20)` | grupo con luminancia (V) media más baja |
| `team_a` | amarillo `(46, 190, 240)` | de los dos grupos restantes, el de **mayor saturación** (desempate: hue menor) |
| `team_b` | azul `(235, 160, 60)` | el otro grupo |
| `unknown` | gris `(180, 180, 180)` | trayectoria con pocas muestras aún |

Etiquetado determinista: mismo vídeo → mismas etiquetas. Los tres colores son configurables.

## 2. Requisitos funcionales

- **RF-1** Para cada detección se extrae el color del torso: recorte del 5 %–45 % de la altura y 25 %–75 %
  del ancho de la caja, en HSV. Cajas de altura < `min_box_height` (40 px) no aportan muestra.
- **RF-2** Cada `track_id` mantiene una media exponencial (EMA) de sus muestras de color.
- **RF-3** Cada `classify_every` frames (def. 30) se re-ejecuta la clasificación sobre las trayectorias con al
  menos `min_color_samples` muestras (def. 12):
  k-means (k=3, semilla fija, distancia euclídea en el espacio `[cos h, sin h, S, V]`) →
  `referee` = centro con menor V; `team_a`/`team_b` = ordenados por saturación descendente
  (desempate por hue ascendente). El orden por saturación es más estable que el hue puro para
  camisetas blancas, cuyo hue es ruido.
- **RF-4** La etiqueta de una trayectoria es **pegajosa**: una vez asignada no cambia de grupo;
  antes de tener muestras suficientes es `unknown`.
- **RF-5** Si no hay 3 grupos separables (menos de 3 trayectorias con muestras), todo queda `unknown`
  y el vídeo se colorea con la paleta v1.
- **RF-6** `detections.json` añade `"team"` a cada detección y `tracks.json` añade `"team"` y `"mean_hsv"`
  a cada trayectoria.
- **RF-7** `--no-teams` desactiva la clasificación (comportamiento v1: color por `track_id`).
- **RF-8** Colores y umbrales configurables en `config/default.yaml`
  (`teams.classify`, `teams.min_color_samples`, `teams.classify_every`, `teams.min_box_height`, `teams.colors`).
- **RF-9** El resumen final imprime el recuento de trayectorias por etiqueta.

## 3. Requisitos no funcionales

- Sin dependencias nuevas (k-means propio en NumPy, ~30 líneas, determinista).
- Coste adicional < 3 % del tiempo total (un `cvtColor` + media por caja).
- Tests unitarios con fotogramas sintéticos (camisetas de color conocido).

## 4. Criterios de aceptación

1. `--video videos/okliga_prueba.mp4 --max-seconds 20` produce cajas en **tres colores** distinguibles en el MP4.
2. `tracks.json` asigna `referee` a exactamente un grupo y `team_a`/`team_b` al resto clasificado;
   los grupos son disjuntos.
3. El grupo `referee` tiene `mean_hsv` con V estrictamente menor que la media de `team_a` y `team_b`.
4. `--no-teams` reproduce el comportamiento de la v1 (color por `track_id`).
5. Repetir la misma ejecución produce las mismas etiquetas por `track_id` (determinismo).
6. `pytest`, `ruff` y `mypy` en verde; los tests nuevos cubren muestreo, etiquetado y determinismo.

## 5. Limitaciones

- Jugadores de espaldas con dorsal grande, o con camiseta del color del fondo, pueden clasificarse mal.
- Un equipo con camiseta muy oscura (azul marino) puede confundirse con el árbitro negro.
  Mitigación: `teams.referee_override` no implementado en v2; se puede forzar en v3 con mockup manual.
- Los primeros frames de cada trayectoria salen en gris hasta acumular `min_color_samples`.

## 6. Resultado medido (okliga_prueba.mp4, 20 s)

| Etiqueta | Trayectorias | H medio | S medio | V medio |
|---|---|---|---|---|
| `referee` | 12 | 116.7 | 67.9 | **60.8** |
| `team_a` (amarillo) | 13 | 57.3 | 130.5 | 143.2 |
| `team_b` (blanco) | 16 | 67.5 | 85.0 | 151.1 |

El árbitro queda separado por un margen de V > 80 puntos. Coste añadido: ~1 % del tiempo total
(33.0 fps con equipos vs 33.4 fps sin ellos).
