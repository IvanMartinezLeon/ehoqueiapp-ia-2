# SPEC — Revisión y limpieza de pseudo-etiquetas

Estado: aprobada por el usuario (2025, tras pre-etiquetar `videos/temp.mp4`).

## Problema

`autolabel.py` a `--conf 0.10` maximiza recall pero mete ruido: en las 63 imágenes de `partido_*`
salieron 931 cajas con cajas de hasta 1288 px de alto (público, barandilla, cajas fusionadas) y
hasta 5 porteros en un mismo frame. Etiquetar a mano 63 imágenes a 4K sin ayuda es lento, y revisar
una a una tampoco es viable.

## Objetivo

Dos herramientas de apoyo al etiquetado humano, sin decidir semántica por nosotros:

1. **Hojas de contacto anotadas** (`scripts/review_labels.py`): renderiza las cajas existentes
   sobre los frames, con color por clase y nombre de fichero, en hojas 4×4. Permite revisar 63
   imágenes en 4 pantallas en lugar de 63.
2. **Limpieza heurística** (`scripts/clean_labels.py`): elimina cajas imposibles por geometría,
   con `--dry-run` por defecto y backup `.bak` antes de escribir.

## Reglas de limpieza (solo geometría, no semántica)

| Regla | Umbral por defecto | Motivo |
|---|---|---|
| Altura mínima | 60 px | cajas minúsculas del público/fondo |
| Altura máxima | 600 px | cajas fusionadas o público pegado a cámara |
| Máx. porteros por frame | 2 | en pista solo hay 2 porteros; el resto son errores de clase |

Se conservan las cajas de mayor área cuando una clase supera su cupo. Nunca se reescribe una caja
ni se cambia su clase: solo se descarta la línea completa.

## No objetivos

- Reasignar `player` → `goalkeeper`/`referee` (requiere criterio visual humano).
- Decidir calidad de una caja más allá de su geometría.
- Tocar `dataset/labels/raw` por defecto: `clean_labels.py` exige `--write` explícito.

## Criterios de aceptación

- `review_labels.py` genera N hojas JPG con todas las cajas y el nombre del frame.
- `clean_labels.py` sin `--write` no escribe nada e imprime el recuento por regla y por clase.
- Con `--write` crea `<label>.txt.bak` la primera vez y deja el `.txt` limpio.
- Tests unitarios de las reglas (altura mín./máx., cupo de porteros, orden estable).

## Resultado medido

Pendiente de la primera ejecución sobre `partido_*` (ver `docs/TASKS.md`, T67–T68).

## Resultado medido

Sobre `partido_*` (63 frames 4K, pre-etiquetado `train-2 @1280 conf 0.10` + edición manual previa):

| Momento | Cajas | player | goalkeeper | referee |
|---|---|---|---|---|
| Tras autolabel | 931 | 761 | 91 | 79 |
| Tras edición manual en labelImg | 873 | 707 | 87 | 79 |
| Tras `clean_labels.py --write` | **750** | 607 | 65 | 78 |

Descartadas: 95 grandes (>600 px), 7 pequeñas (<60 px), 21 porteros sobrantes. 48 backups `.txt.bak`.
