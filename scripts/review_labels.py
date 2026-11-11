#!/usr/bin/env python3
"""Render annotated contact sheets so labelled frames can be reviewed quickly.

Draws every existing YOLO box on a downscaled copy of the image, coloured by class, and tiles
several frames per sheet with the file name on top. Reviewing four sheets is faster (and less
error prone) than opening 60 full-resolution images one by one.

Examples:
    python scripts/review_labels.py --pattern "partido_*.jpg"
    python scripts/review_labels.py --pattern "partido_*.jpg" --per-sheet 12 --width 720
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_COLOURS = {0: (60, 60, 255), 1: (255, 190, 0), 2: (60, 220, 60)}
UNKNOWN_COLOUR = (200, 200, 200)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera hojas de contacto con las cajas etiquetadas dibujadas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", type=Path, default=Path("dataset/images/raw"), help="carpeta de imágenes"
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("dataset/labels/raw"), help="carpeta de etiquetas"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/label_review"), help="carpeta de salida"
    )
    parser.add_argument("--pattern", default="*.jpg", help="glob de imágenes a revisar")
    parser.add_argument("--per-sheet", type=int, default=16, help="frames por hoja")
    parser.add_argument("--columns", type=int, default=4, help="columnas por hoja")
    parser.add_argument("--width", type=int, default=960, help="ancho de cada frame en la hoja")
    parser.add_argument("--thickness", type=int, default=2, help="grosor de las cajas")
    parser.add_argument(
        "--draw-names", action="store_true", default=True, help="escribir el nombre del frame"
    )
    parser.add_argument("--no-draw-names", dest="draw_names", action="store_false")
    return parser.parse_args(argv)


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file as ``(class_id, cx, cy, w, h)`` tuples in normalised units."""
    if not label_path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError:
            continue
        boxes.append((class_id, cx, cy, box_width, box_height))
    return boxes


def draw_boxes(
    frame: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw normalised YOLO boxes on ``frame`` (in place) with their class colour."""
    height, width = frame.shape[:2]
    for class_id, cx, cy, box_width, box_height in boxes:
        x1 = int((cx - box_width / 2.0) * width)
        y1 = int((cy - box_height / 2.0) * height)
        x2 = int((cx + box_width / 2.0) * width)
        y2 = int((cy + box_height / 2.0) * height)
        colour = CLASS_COLOURS.get(class_id, UNKNOWN_COLOUR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    return frame


def build_sheet(
    tiles: list[np.ndarray],
    columns: int,
    label_height: int = 26,
) -> np.ndarray:
    """Tile equally sized frames into one sheet filled with white."""
    if not tiles:
        raise ValueError("no hay frames para la hoja")
    tile_height, tile_width = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full(
        (rows * (tile_height + label_height), columns * tile_width, 3), 255, dtype=np.uint8
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        top = row * (tile_height + label_height)
        left = column * tile_width
        sheet[top : top + tile_height, left : left + tile_width] = tile
    return sheet


def review(args: argparse.Namespace) -> int:
    if not args.images.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de imágenes: {args.images}")
    if args.per_sheet <= 0 or args.columns <= 0:
        raise ValueError("--per-sheet y --columns deben ser positivos")
    images = sorted(
        path for path in args.images.glob(args.pattern) if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no hay imágenes con patrón {args.pattern} en {args.images}")

    rows = max(1, args.per_sheet // args.columns)
    cell_height = max(1, round(args.width * 9 / 16))
    label_height = 26 if args.draw_names else 0
    args.output.mkdir(parents=True, exist_ok=True)

    sheets = 0
    missing = 0
    total_boxes = 0
    for start in range(0, len(images), args.per_sheet):
        chunk = images[start : start + args.per_sheet]
        tiles: list[np.ndarray] = []
        for image_path in chunk:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"  aviso: no se puede leer {image_path.name}")
                continue
            boxes = read_boxes(args.labels / f"{image_path.stem}.txt")
            if not boxes:
                missing += 1
            total_boxes += len(boxes)
            small = cv2.resize(frame, (args.width, cell_height))
            draw_boxes(small, boxes, args.thickness)
            tile = np.full((cell_height + label_height, args.width, 3), 255, dtype=np.uint8)
            tile[label_height:] = small
            if args.draw_names:
                text = f"{image_path.name}  ({len(boxes)} cajas)"
                cv2.putText(
                    tile,
                    text,
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            tiles.append(tile)
        if not tiles:
            continue
        sheets += 1
        sheet_path = args.output / f"sheet_{sheets:03d}.jpg"
        cv2.imwrite(
            str(sheet_path),
            build_sheet(tiles, args.columns, label_height),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        print(f"  {sheet_path}  ({len(tiles)} frames)")

    print("\n--- Resumen ---")
    print(f"Imágenes revisadas : {len(images)}")
    print(f"Hojas generadas    : {sheets} (filas por hoja: {rows})")
    print(f"Cajas dibujadas    : {total_boxes}")
    print(f"Sin etiqueta       : {missing}")
    print(f"Carpeta            : {args.output.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return review(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
