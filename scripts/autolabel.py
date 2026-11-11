#!/usr/bin/env python3
"""Pre-annotate images with a COCO person detector so manual labelling is faster.

For every image it writes a YOLO label file with the detected people as a single class
(``player`` by default). You then only have to fix what needs judgement: turn goalkeepers and
referees into their own class, delete false positives and add the ball.

Existing label files are respected (``--skip-existing`` is on by default) so your manual work is
never overwritten.

With a model trained on this project (``player=0, goalkeeper=1, referee=2``) use ``--classes 0,1,2
--keep-classes`` so every box keeps the class the model predicted instead of collapsing to
``--class-id``.

Examples:
    python scripts/autolabel.py                          # dataset/images/raw -> dataset/labels/raw
    python scripts/autolabel.py --conf 0.30 --class-id 0
    python scripts/autolabel.py --overwrite              # regenerate everything
    python scripts/autolabel.py --model runs/detect/train-3/weights/best.pt \
        --classes 0,1,2 --keep-classes --imgsz 1280    # etiquetas multi-clase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo-etiqueta imágenes con un detector de personas para acelerar el etiquetado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", type=Path, default=Path("dataset/images/raw"), help="carpeta de imágenes"
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("dataset/labels/raw"), help="carpeta de etiquetas"
    )
    parser.add_argument(
        "--model", default="yolo11m.pt", help="pesos del detector (clase person de COCO)"
    )
    parser.add_argument("--conf", type=float, default=0.35, help="confianza mínima")
    parser.add_argument("--imgsz", type=int, default=960, help="resolución de inferencia")
    parser.add_argument(
        "--class-id", type=int, default=0, help="índice de clase a escribir en el .txt"
    )
    parser.add_argument(
        "--classes",
        default="0",
        help="clases del modelo a conservar, separadas por comas (COCO person = 0)",
    )
    parser.add_argument(
        "--keep-classes",
        action="store_true",
        help="escribir la clase predicha por el modelo en vez de --class-id",
    )
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument(
        "--min-height",
        type=int,
        default=25,
        help="descartar cajas más bajas que estos píxeles (falsos positivos del público)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="sobrescribir etiquetas existentes"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="mostrar lo que se haría sin escribir ficheros"
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:  # pragma: no cover - torch is a hard dependency of ultralytics
        pass
    return "cpu"


def parse_class_ids(raw: str) -> list[int]:
    """Parse a comma separated list of model class indices (``"0,1,2"``)."""
    ids = [int(part) for part in raw.replace(" ", "").split(",") if part]
    if not ids:
        raise ValueError("--classes no puede estar vacío")
    return ids


def boxes_to_yolo(
    boxes: list[tuple[float, float, float, float, float]],
    width: int,
    height: int,
    class_ids: list[int],
    min_height: int,
) -> list[str]:
    """Convert (x1, y1, x2, y2, conf) pixel boxes into normalised YOLO lines."""
    if len(class_ids) != len(boxes):
        raise ValueError("class_ids debe tener una entrada por caja")
    lines: list[str] = []
    for (x1, y1, x2, y2, _conf), class_id in zip(boxes, class_ids, strict=True):
        if (y2 - y1) < min_height or (x2 - x1) < 3:
            continue
        cx = max(min((x1 + x2) / 2.0 / width, 1.0), 0.0)
        cy = max(min((y1 + y2) / 2.0 / height, 1.0), 0.0)
        box_width = max(min((x2 - x1) / width, 1.0), 0.0)
        box_height = max(min((y2 - y1) / height, 1.0), 0.0)
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {box_width:.6f} {box_height:.6f}")
    return lines


def autolabel(args: argparse.Namespace) -> int:
    if not 0.0 < args.conf < 1.0:
        raise ValueError("--conf debe estar entre 0 y 1")
    model_classes = parse_class_ids(args.classes)
    if not args.images.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de imágenes: {args.images}")
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no hay imágenes en {args.images}")

    pending = [
        path for path in images if args.overwrite or not (args.labels / f"{path.stem}.txt").exists()
    ]
    skipped = len(images) - len(pending)
    print(f"Imágenes: {len(images)} | a procesar: {len(pending)} | ya etiquetadas: {skipped}")
    if not pending:
        print("nada que hacer: todas las imágenes tienen ya etiqueta")
        return 0
    if args.dry_run:
        print("dry-run: no se escribe nada")
        return 0

    from ultralytics import YOLO  # imported late so --help stays instant
    from ultralytics.engine.results import Results

    args.labels.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    device = resolve_device(args.device)
    print(
        f"Modelo {args.model} en {device} (imgsz {args.imgsz}, conf {args.conf}, "
        f"clases {','.join(str(c) for c in model_classes)})"
    )

    written = 0
    empty = 0
    total_boxes = 0
    for image_path in pending:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  aviso: no se puede leer {image_path.name}")
            continue
        height, width = frame.shape[:2]
        predictions = list(
            model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                classes=model_classes,
                device=device,
                verbose=False,
            )
        )
        if not predictions or not isinstance(predictions[0], Results):
            print(f"  aviso: predicción inesperada en {image_path.name}")
            continue
        detected = predictions[0].boxes
        if detected is None or len(detected) == 0:
            empty += 1
            continue
        data = detected.cpu().numpy()
        boxes = [
            (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(conf))
            for row, conf in zip(data.xyxy, data.conf, strict=True)
        ]
        class_ids = (
            [int(cls) for cls in data.cls] if args.keep_classes else [args.class_id] * len(boxes)
        )
        lines = boxes_to_yolo(boxes, width, height, class_ids, args.min_height)
        if not lines:
            empty += 1
            continue
        label_path = args.labels / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
        total_boxes += len(lines)

    print("\n--- Resumen ---")
    print(f"Etiquetas escritas      : {written}")
    print(f"Cajas generadas         : {total_boxes}")
    print(f"Sin detecciones         : {empty} (se dejan sin etiquetar)")
    print(f"Ya etiquetadas (omitidas): {skipped}")
    print(f"Carpeta                 : {args.labels.resolve()}")
    print("\nSiguiente paso: abre ./scripts/label.sh y corrige clases (portero/árbitro),")
    print("borra falsos positivos del público y añade el balón.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return autolabel(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
