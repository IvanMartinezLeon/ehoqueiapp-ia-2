#!/usr/bin/env python3
"""Split a labelled image folder into YOLO train/val sets and write ``data.yaml``.

Expected input (YOLO format):

    dataset/images/raw/frame_000001.jpg
    dataset/labels/raw/frame_000001.txt      # one line per box: "class cx cy w h" (normalised)

Output:

    dataset/images/train, dataset/images/val
    dataset/labels/train, dataset/labels/val
    dataset/data.yaml

Images without a label file are reported and skipped by default (unlabelled images look like
background to YOLO and hurt training).

Examples:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --check-only     # validate, write nothing
    python scripts/split_dataset.py --val-ratio 0.25 --seed 7
    python scripts/split_dataset.py --classes player,goalkeeper,referee,ball
    python scripts/split_dataset.py --move          # move instead of copy
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ("player", "goalkeeper", "referee", "ball")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide un dataset etiquetado en train/val y genera data.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", type=Path, default=Path("dataset/images/raw"), help="JPG de entrada"
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("dataset/labels/raw"), help="TXT de entrada"
    )
    parser.add_argument("--output", type=Path, default=Path("dataset"), help="raíz del dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="fracción para validación")
    parser.add_argument("--seed", type=int, default=0, help="semilla del reparto aleatorio (reproducible)")
    parser.add_argument(
        "--group-by",
        choices=("prefix", "random"),
        default="prefix",
        help="prefix: separa por vídeo y por tiempo (evita frames gemelos en train y val); random: azar",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help=(
            "nombres de clase separados por comas, en orden de índice YOLO "
            f"(default: los de data.yaml si existe, si no {','.join(DEFAULT_CLASSES)})"
        ),
    )
    parser.add_argument(
        "--min-instances",
        type=int,
        default=50,
        help="mínimo de cajas por clase antes de avisar (0 desactiva el aviso)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validar y mostrar el resumen sin copiar nada ni escribir data.yaml",
    )
    parser.add_argument("--move", action="store_true", help="mover en vez de copiar")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="vaciar images/train, images/val, labels/train y labels/val antes de repartir",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="incluir imágenes sin fichero de etiquetas (no recomendado)",
    )
    return parser.parse_args(argv)


def group_of(stem: str) -> str:
    """Group key of an image: the prefix before the trailing "_<number>" (``noa_000010`` -> ``noa``)."""
    prefix, _, suffix = stem.rpartition("_")
    if prefix and suffix.isdigit():
        return prefix
    return stem


def split_by_group(
    items: list[tuple[Path, Path]], val_ratio: float, min_group_size: int = 5
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split so that frames of the same video never land on both sides of the fence.

    Frames extracted every N seconds are near-duplicates: a random split puts twins in train and val
    and inflates the metrics. Groups are split by time (the tail of each clip goes to validation).
    Groups smaller than ``min_group_size`` stay entirely in train.
    """
    buckets: dict[str, list[tuple[Path, Path]]] = {}
    for image, label in items:
        buckets.setdefault(group_of(image.stem), []).append((image, label))

    train: list[tuple[Path, Path]] = []
    val: list[tuple[Path, Path]] = []
    for group in sorted(buckets):
        members = sorted(buckets[group], key=lambda pair: _frame_number(pair[0].stem))
        if len(members) < min_group_size:
            train.extend(members)
            continue
        count = min(max(round(len(members) * val_ratio), 1), len(members) - 1)
        train.extend(members[:-count])
        val.extend(members[-count:])
    return train, val


def _frame_number(stem: str) -> int:
    suffix = stem.rpartition("_")[2]
    return int(suffix) if suffix.isdigit() else 0


def read_classes(raw: str) -> list[str]:
    classes = [name.strip() for name in raw.split(",") if name.strip()]
    if not classes:
        raise ValueError("la lista de clases está vacía")
    return classes


def read_classes_from_yaml(output: Path) -> list[str] | None:
    """Return the class names already declared in ``<output>/data.yaml``, if any."""
    path = output / "data.yaml"
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    names = payload.get("names")
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return None


def resolve_classes(args: argparse.Namespace) -> list[str]:
    """Explicit ``--classes`` wins, then ``data.yaml``, then the built-in default."""
    if args.classes:
        return read_classes(args.classes)
    inherited = read_classes_from_yaml(args.output)
    if inherited:
        print(f"Clases tomadas de {args.output / 'data.yaml'}: {', '.join(inherited)}")
        return inherited
    return list(DEFAULT_CLASSES)


def validate_label_file(path: Path, class_count: int) -> tuple[bool, int, int]:
    """Check YOLO label syntax. Returns (is_valid, box_count, class_id_max)."""
    if not path.exists():
        return False, 0, -1
    boxes = 0
    highest = -1
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            print(f"  aviso: {path.name}:{line_number} tiene {len(parts)} columnas (se esperan 5)")
            return False, 0, -1
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"  aviso: {path.name}:{line_number} no es numérico")
            return False, 0, -1
        if not 0 <= class_id < class_count:
            print(f"  aviso: {path.name}:{line_number} clase {class_id} fuera de rango")
            return False, 0, -1
        if any(not 0.0 <= value <= 1.0 for value in values):
            print(f"  aviso: {path.name}:{line_number} coordenadas fuera de [0, 1]")
            return False, 0, -1
        boxes += 1
        highest = max(highest, class_id)
    if boxes == 0:
        return False, 0, -1
    return True, boxes, highest


def transfer(source: Path, destination: Path, move: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)
    except OSError as error:
        raise RuntimeError(f"no se pudo copiar/mover {source} -> {destination}: {error}") from error


def write_data_yaml(output: Path, classes: list[str]) -> Path:
    payload = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": {index: name for index, name in enumerate(classes)},
    }
    path = output / "data.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


@dataclass(slots=True)
class ScanResult:
    """Validated contents of the raw image/label folders."""

    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    unlabeled: list[Path] = field(default_factory=list)
    invalid: list[Path] = field(default_factory=list)
    total_boxes: int = 0
    histogram: dict[int, int] = field(default_factory=dict)


def scan(args: argparse.Namespace, classes: list[str]) -> ScanResult:
    """Walk the raw folders and collect every valid ``(image, label)`` pair."""
    if not args.images.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de imágenes: {args.images}")

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no hay imágenes en {args.images}")

    result = ScanResult()
    print(f"Revisando {len(images)} imágenes...")
    for image in images:
        label = args.labels / f"{image.stem}.txt"
        if not label.exists():
            result.unlabeled.append(image)
            continue
        is_valid, boxes, _ = validate_label_file(label, len(classes))
        if not is_valid:
            result.invalid.append(label)
            continue
        result.pairs.append((image, label))
        result.total_boxes += boxes
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                result.histogram[class_id] = result.histogram.get(class_id, 0) + 1

    if args.include_unlabeled:
        result.pairs.extend((image, Path()) for image in result.unlabeled)
    return result


def partition(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    seed: int,
    group_by: str = "random",
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Deterministic ``(train, val)`` split so ``--check-only`` previews the real one.

    ``group_by="prefix"`` keeps every video on one side of the fence and splits each clip by time;
    ``"random"`` shuffles individual frames (near-duplicates then leak into validation).
    """
    if group_by == "prefix":
        return split_by_group(pairs, val_ratio)
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(round(len(shuffled) * val_ratio), 1)
    return shuffled[val_count:], shuffled[:val_count]


def report(
    result: ScanResult,
    classes: list[str],
    train_count: int,
    val_count: int,
    data_yaml: Path | None,
    min_instances: int,
    include_unlabeled: bool,
) -> tuple[list[str], list[str]]:
    """Print the summary. Returns ``(failures, notes)``: only failures block a check."""
    print("\n--- Resumen ---")
    print(f"Clases                  : {', '.join(classes)}")
    print(f"Imágenes válidas        : {len(result.pairs)}")
    print(f"  train                 : {train_count}")
    print(f"  val                   : {val_count}")
    print(f"Cajas etiquetadas       : {result.total_boxes}")
    print(
        f"Imágenes sin etiqueta   : {len(result.unlabeled)}"
        + (" (incluidas)" if include_unlabeled else " (omitidas)")
    )
    print(f"Etiquetas inválidas     : {len(result.invalid)}")
    distribution = ", ".join(
        f"{classes[index]}={count}" for index, count in sorted(result.histogram.items())
    )
    print(f"Reparto por clase       : {distribution or '-'}")
    if data_yaml is not None:
        print(f"data.yaml               : {data_yaml}")

    failures: list[str] = []
    notes: list[str] = []
    if result.invalid:
        failures.append(f"{len(result.invalid)} etiqueta(s) inválida(s); mira los avisos de arriba")
    if result.unlabeled and not include_unlabeled:
        notes.append(
            f"{len(result.unlabeled)} imagen(es) sin etiqueta omitida(s); usa "
            "--include-unlabeled solo si son negativos a propósito"
        )
    if train_count < 20:
        failures.append(f"solo {train_count} imágenes de train; el modelo no aprenderá mucho")
    for index, name in enumerate(classes):
        count = result.histogram.get(index, 0)
        if count == 0:
            failures.append(f"clase '{name}' sin ninguna caja")
        elif min_instances > 0 and count < min_instances:
            failures.append(
                f"clase '{name}' con {count} caja(s) por debajo del umbral {min_instances}; "
                "añade imágenes o saldrá con mAP bajo"
            )
    return failures, notes


def split(args: argparse.Namespace, classes: list[str]) -> int:
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio debe estar entre 0 y 1")

    result = scan(args, classes)
    if len(result.pairs) < 2:
        print(
            f"error: solo {len(result.pairs)} imagen(es) etiquetadas válidas; "
            "hacen falta al menos 2",
            file=sys.stderr,
        )
        print(
            f"  sin etiqueta: {len(result.unlabeled)} | inválidas: {len(result.invalid)}",
            file=sys.stderr,
        )
        return 1

    if args.clean:
        for subset in ("train", "val"):
            for folder in (args.output / "images" / subset, args.output / "labels" / subset):
                if folder.is_dir():
                    for stale in folder.iterdir():
                        if stale.is_file():
                            stale.unlink()
                    print(f"limpiado: {folder}")

    train_set, val_set = partition(result.pairs, args.val_ratio, args.seed, args.group_by)
    if args.group_by == "prefix" and not args.check_only:
        counts: dict[str, list[int]] = {}
        for subset, pairs in (("train", train_set), ("val", val_set)):
            for image, _label in pairs:
                bucket = counts.setdefault(group_of(image.stem), [0, 0])
                bucket[0 if subset == "train" else 1] += 1
        print(
            "Reparto por vídeo (t/v) : "
            + ", ".join(f"{name} {t}/{v}" for name, (t, v) in sorted(counts.items()))
        )
    for subset, pairs in (("train", train_set), ("val", val_set)):
        for image, label in pairs:
            transfer(image, args.output / "images" / subset / image.name, args.move)
            if label.name:
                transfer(label, args.output / "labels" / subset / label.name, args.move)

    data_yaml = write_data_yaml(args.output, classes)
    failures, notes = report(
        result,
        classes,
        len(train_set),
        len(val_set),
        data_yaml,
        args.min_instances,
        args.include_unlabeled,
    )
    for note in notes:
        print(f"nota : {note}")
    for failure in failures:
        print(f"aviso: {failure}")
    return 0


def check(args: argparse.Namespace, classes: list[str]) -> int:
    """Validate the raw folders and exit non-zero on any blocking problem."""
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio debe estar entre 0 y 1")

    result = scan(args, classes)
    if len(result.pairs) < 2:
        print(
            f"error: solo {len(result.pairs)} imagen(es) etiquetadas válidas; "
            "hacen falta al menos 2",
            file=sys.stderr,
        )
        return 1

    train_set, val_set = partition(result.pairs, args.val_ratio, args.seed, args.group_by)
    if args.group_by == "prefix" and not args.check_only:
        counts: dict[str, list[int]] = {}
        for subset, pairs in (("train", train_set), ("val", val_set)):
            for image, _label in pairs:
                bucket = counts.setdefault(group_of(image.stem), [0, 0])
                bucket[0 if subset == "train" else 1] += 1
        print(
            "Reparto por vídeo (t/v) : "
            + ", ".join(f"{name} {t}/{v}" for name, (t, v) in sorted(counts.items()))
        )
    failures, notes = report(
        result,
        classes,
        len(train_set),
        len(val_set),
        None,
        args.min_instances,
        args.include_unlabeled,
    )
    for note in notes:
        print(f"nota : {note}")
    for failure in failures:
        print(f"fallo: {failure}")
    if failures:
        print(f"\nCheck fallido: {len(failures)} problema(s).")
        return 1
    print("\nCheck OK. Nada escrito.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        classes = resolve_classes(args)
        if args.check_only:
            return check(args, classes)
        return split(args, classes)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
