#!/usr/bin/env python3
"""Flag the darkest-kit person in each frame as a referee candidate.

Referees wear dark kits, so this script looks for the least bright shirt among the boxes of every
image and suggests it as ``class 2`` (``referee``). Goalkeepers often wear dark kits too, so a
candidate is a *hint for review*, not an answer: check the images this prints instead of reviewing
all of them.

Default mode is a dry run (nothing is written); pass ``--apply`` to rewrite the label files.

Examples:
    python scripts/guess_referee.py                       # show candidates only
    python scripts/guess_referee.py --apply               # write class 2 on candidates
    python scripts/guess_referee.py --max-value 105       # stricter (darker) threshold
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from player_detect.teams import sample_jersey_color

REFEREE_CLASS_ID = 2
MIN_BOX_HEIGHT = 20
AMBIGUITY_MARGIN = 15.0


@dataclass
class Candidate:
    image: str
    line_index: int
    value: float
    runner_up: float | None
    lines: list[str]

    @property
    def ambiguous(self) -> bool:
        return self.runner_up is not None and (self.runner_up - self.value) < AMBIGUITY_MARGIN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sugiere el árbitro por color de camiseta (revisión, no verdad absoluta).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", type=Path, default=Path("dataset/images/raw"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels/raw"))
    parser.add_argument("--class-id", type=int, default=REFEREE_CLASS_ID, help="clase a escribir")
    parser.add_argument(
        "--max-value",
        type=float,
        default=115.0,
        help="luminancia máxima (HSV V) para considerar la camiseta oscura",
    )
    parser.add_argument("--apply", action="store_true", help="escribir los cambios en los .txt")
    return parser.parse_args(argv)


def find_candidate(image_path: Path, label_path: Path, max_value: float) -> Candidate | None:
    """Return the darkest-kit box of the image, or None when nobody looks like a referee."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    height, width = frame.shape[:2]
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    values: list[tuple[int, float]] = []
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            continue
        _cls, cx, cy, box_w, box_h = (float(value) for value in parts)
        bbox = (
            (cx - box_w / 2) * width,
            (cy - box_h / 2) * height,
            (cx + box_w / 2) * width,
            (cy + box_h / 2) * height,
        )
        feature = sample_jersey_color(frame, bbox, min_box_height=MIN_BOX_HEIGHT)
        if feature is not None:
            values.append((index, feature.value))
    if not values:
        return None
    values.sort(key=lambda pair: pair[1])
    best_index, best_value = values[0]
    if best_value > max_value:
        return None
    runner_up = values[1][1] if len(values) > 1 else None
    return Candidate(
        image=image_path.name,
        line_index=best_index,
        value=best_value,
        runner_up=runner_up,
        lines=lines,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.images.is_dir() or not args.labels.is_dir():
        print("error: no existen las carpetas de imágenes o etiquetas", file=sys.stderr)
        return 2
    label_files = sorted(args.labels.glob("frame_*.txt"))
    if not label_files:
        print("error: no hay etiquetas que revisar", file=sys.stderr)
        return 2

    candidates: list[Candidate] = []
    for label_path in label_files:
        image_path = args.images / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        candidate = find_candidate(image_path, label_path, args.max_value)
        if candidate is not None:
            candidates.append(candidate)

    ambiguous = [c for c in candidates if c.ambiguous]
    print(f"Etiquetas revisadas      : {len(label_files)}")
    print(f"Candidatos a árbitro     : {len(candidates)}")
    print(
        f"  de ellos ambiguos      : {len(ambiguous)} (segunda camiseta oscura a <{AMBIGUITY_MARGIN:.0f} de V)"
    )
    print("\nImágenes a revisar (el portero también viste oscuro; decide tú):")
    for candidate in candidates:
        flag = "  ¿portero?" if candidate.ambiguous else ""
        runner = f" | 2º V={candidate.runner_up:.0f}" if candidate.runner_up is not None else ""
        print(f"  {candidate.image}  V={candidate.value:.0f}{runner}{flag}")

    if not args.apply:
        print("\nDry-run: no se ha escrito nada. Añade --apply para marcar la clase en los .txt.")
        return 0

    changed = 0
    for candidate in candidates:
        label_path = args.labels / f"{Path(candidate.image).stem}.txt"
        lines = candidate.lines[:]
        parts = lines[candidate.line_index].split()
        if int(float(parts[0])) == args.class_id:
            continue
        lines[candidate.line_index] = " ".join([str(args.class_id), *parts[1:]])
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        changed += 1
    print(f"\nEtiquetas modificadas    : {changed}")
    print("Revisa esas imágenes en labelImg y corrige lo que sea portero u otro jugador.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
