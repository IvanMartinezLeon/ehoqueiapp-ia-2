#!/usr/bin/env python3
"""Remove geometrically impossible boxes from YOLO label files (safe, dry-run by default).

The pre-annotator maximises recall, so it also produces boxes that cannot be a player: huge boxes
that merge a player with the barrier or the crowd, tiny boxes from the stands, and more
goalkeepers than the two that exist on the rink. This script drops those lines by pure geometry —
it never moves a box and never changes its class.

Dry-run is the default: nothing is written unless ``--write`` is passed, and the first write keeps
the original as ``<name>.txt.bak``.

Examples:
    python scripts/clean_labels.py --pattern "partido_*.txt"              # solo informe
    python scripts/clean_labels.py --pattern "partido_*.txt" --write      # aplica y hace backup
    python scripts/clean_labels.py --min-height 50 --max-height 900 --max-goalkeepers 2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee"}
GOALKEEPER_ID = 1


@dataclass(frozen=True)
class CleanRules:
    """Geometric limits applied to every label line."""

    min_height_px: float = 60.0
    max_height_px: float = 600.0
    max_goalkeepers: int = 2
    dropped_classes: tuple[int, ...] = ()


@dataclass(frozen=True)
class Box:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float
    line: str = field(compare=False, default="")

    def area(self) -> float:
        return self.width * self.height


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Elimina cajas imposibles por geometría en ficheros de etiquetas YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("dataset/labels/raw"), help="carpeta de etiquetas"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("dataset/images/raw"),
        help="carpeta de imágenes (para conocer el tamaño real del frame)",
    )
    parser.add_argument("--pattern", default="*.txt", help="glob de etiquetas a limpiar")
    parser.add_argument(
        "--width", type=int, default=0, help="ancho en px si no se encuentran las imágenes"
    )
    parser.add_argument(
        "--height", type=int, default=0, help="alto en px si no se encuentran las imágenes"
    )
    parser.add_argument(
        "--min-height", type=float, default=60.0, help="altura mínima de caja en píxeles"
    )
    parser.add_argument(
        "--max-height", type=float, default=600.0, help="altura máxima de caja en píxeles"
    )
    parser.add_argument(
        "--max-goalkeepers",
        type=int,
        default=2,
        help="porteros máximos por frame (0 desactiva el cupo)",
    )
    parser.add_argument(
        "--drop-class",
        default="",
        help="índices de clase a borrar por completo, separados por comas (p. ej. 3 para ball)",
    )
    parser.add_argument(
        "--write", action="store_true", help="aplicar los cambios (por defecto solo informe)"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="no crear <nombre>.txt.bak al escribir"
    )
    return parser.parse_args(argv)


def parse_boxes(lines: list[str]) -> list[Box]:
    """Parse YOLO lines, ignoring malformed ones."""
    boxes: list[Box] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            continue
        boxes.append(Box(class_id, cx, cy, width, height, line=line))
    return boxes


def clean_boxes(
    boxes: list[Box],
    frame_width: int,
    frame_height: int,
    rules: CleanRules,
) -> tuple[list[Box], dict[str, int]]:
    """Return the boxes that survive ``rules`` plus a per-reason removal count."""
    removed = {"small": 0, "large": 0, "extra_goalkeeper": 0, "class": 0}
    kept: list[Box] = []
    for box in boxes:
        if box.class_id in rules.dropped_classes:
            removed["class"] += 1
            continue
        height_px = box.height * frame_height
        if height_px < rules.min_height_px:
            removed["small"] += 1
            continue
        if height_px > rules.max_height_px:
            removed["large"] += 1
            continue
        kept.append(box)

    if rules.max_goalkeepers > 0:
        goalkeepers = [box for box in kept if box.class_id == GOALKEEPER_ID]
        if len(goalkeepers) > rules.max_goalkeepers:
            keep_ids = {
                id(box)
                for box in sorted(goalkeepers, key=Box.area, reverse=True)[: rules.max_goalkeepers]
            }
            surplus = [box for box in goalkeepers if id(box) not in keep_ids]
            removed["extra_goalkeeper"] = len(surplus)
            surplus_ids = {id(box) for box in surplus}
            kept = [box for box in kept if id(box) not in surplus_ids]

    return kept, removed


def frame_size(
    image_path: Path, fallback_width: int, fallback_height: int
) -> tuple[int, int] | None:
    """Return ``(width, height)`` of the source frame, or the fallback, or None."""
    if image_path.exists():
        frame = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if frame is not None:
            return frame.shape[1] * 8, frame.shape[0] * 8
    if fallback_width > 0 and fallback_height > 0:
        return fallback_width, fallback_height
    return None


def clean(args: argparse.Namespace) -> int:
    if not args.labels.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de etiquetas: {args.labels}")
    rules = CleanRules(
        min_height_px=args.min_height,
        max_height_px=args.max_height,
        max_goalkeepers=args.max_goalkeepers,
        dropped_classes=tuple(
            int(part) for part in args.drop_class.replace(" ", "").split(",") if part
        ),
    )
    labels = sorted(args.labels.glob(args.pattern))
    if not labels:
        raise FileNotFoundError(f"no hay etiquetas con patrón {args.pattern} en {args.labels}")

    totals = {"small": 0, "large": 0, "extra_goalkeeper": 0, "class": 0}
    changed_files = 0
    skipped_no_size = 0
    before_classes: dict[int, int] = {}
    after_classes: dict[int, int] = {}

    for label_path in labels:
        lines = [
            line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        boxes = parse_boxes(lines)
        for box in boxes:
            before_classes[box.class_id] = before_classes.get(box.class_id, 0) + 1

        image_path = next(
            (
                args.images / f"{label_path.stem}{suffix}"
                for suffix in sorted(IMAGE_SUFFIXES)
                if (args.images / f"{label_path.stem}{suffix}").exists()
            ),
            args.images / f"{label_path.stem}.jpg",
        )
        size = frame_size(image_path, args.width, args.height)
        if size is None:
            skipped_no_size += 1
            continue

        kept, removed = clean_boxes(boxes, size[0], size[1], rules)
        for reason, count in removed.items():
            totals[reason] += count
        for box in kept:
            after_classes[box.class_id] = after_classes.get(box.class_id, 0) + 1

        if not any(removed.values()):
            continue
        changed_files += 1
        print(
            f"{label_path.name}: {len(boxes)} -> {len(kept)} cajas "
            f"(pequeñas {removed['small']}, grandes {removed['large']}, "
            f"porteros extra {removed['extra_goalkeeper']}, clase {removed['class']})"
        )
        if args.write:
            if not args.no_backup:
                backup = label_path.with_suffix(label_path.suffix + ".bak")
                if not backup.exists():
                    shutil.copy2(label_path, backup)
            label_path.write_text("\n".join(box.line for box in kept) + "\n", encoding="utf-8")

    print("\n--- Resumen ---")
    print(f"Etiquetas revisadas  : {len(labels)}")
    print(f"Ficheros a limpiar   : {changed_files}")
    print(f"Cajas pequeñas (<{rules.min_height_px:g} px) : {totals['small']}")
    print(f"Cajas grandes (>{rules.max_height_px:g} px) : {totals['large']}")
    print(f"Porteros sobrantes   : {totals['extra_goalkeeper']}")
    print(f"Clase eliminada      : {totals['class']}")
    if skipped_no_size:
        print(f"Sin tamaño de frame  : {skipped_no_size} (usa --width/--height)")
    print(
        "Cajas por clase antes: "
        + ", ".join(f"{CLASS_NAMES.get(k, k)}={v}" for k, v in sorted(before_classes.items()))
    )
    if args.write:
        print(
            "Cajas por clase ahora: "
            + ", ".join(f"{CLASS_NAMES.get(k, k)}={v}" for k, v in sorted(after_classes.items()))
        )
        print("Cambios aplicados (backup .txt.bak creado si no existía).")
    else:
        print("Dry-run: no se ha escrito nada. Repite con --write para aplicar.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return clean(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
