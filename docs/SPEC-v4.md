# SPEC v4 — Anclaje verificado por líneas de pista (anti-deriva)

Estado: **aprobada e implementada**. Sustituye al anclaje de v3 (`AnchorTracker`).

## Problema

Dos estrategias fallaban en `okliga_prueba.mp4` (medido, residual de líneas respecto al frame 0):

| Estrategia | Residual medio | Comportamiento |
|---|---|---|
| Encadenar homografías frame a frame | 48–147 px | Deriva acumulada; el minimapa amontona jugadores en un borde |
| Anclar cada frame al frame 0, sin verificar | 38 px (0.1–193) | Cuando no hay solape no ancla; cuando lo hay a veces se engancha a gradas/LED y devuelve una homografía **con seguridad falsa** |

El fallo de raíz no era el matcher, era **aceptar sin verificar** y **descartar el keyframe de calibración** al rotar la lista de keyframes.

## Solución implementada (`src/player_detect/world.py`)

- `floor_line_mask`: máscara HSV de líneas amarillas del suelo, filtrada por forma (área ≥ 150 px,
  longitud ≥ 70 px, aspecto ≥ 3.5) para descartar camisetas amarillas y manchas.
- `line_residual`: distancia media (px) de las líneas del frame actual a las del frame de calibración
  tras aplicar la homografía candidata. Es la **verificación**.
- `WorldTracker`:
  - El primer keyframe queda **fijado** (ancla absoluta); además mantiene hasta 2 keyframes recientes.
  - Cada `anchor_interval` (5) frames intenta registrar contra los candidatos y **acepta solo si el
    residual de líneas ≤ `world_max_residual_px` (20)**. Si hay varias candidatas, gana el residual menor.
  - Sin líneas visibles, solo acepta un match muy fuerte (`inliers ≥ 120`).
  - Entre anclajes, el transform se arrastra con LK (barato y preciso a corta distancia).

## Resultado medido (600 frames)

| Métrica | Encadenado (v3) | **v4 anclaje verificado** |
|---|---|---|
| Residual mediano | ~48 px | **15.2 px** |
| Residual medio | 147 px (p80) | 27.3 px |
| Matches falsos aceptados | sí | **0** (89 rechazados por verificación) |
| Coste | 0 | 9.1 fps en pipeline (vs 24 fps sin anclaje) |

## Límite conocido

El error restante está dominado por la **calibración de pista**, no por la deriva: el cuadrilátero
actual mapea el suelo visible a 40×20 m y varios jugadores caen a x ≈ −9 m. Con una calibración que
use líneas reales (fondo, área, línea central) el minimapa quedaría dentro de la pista.
