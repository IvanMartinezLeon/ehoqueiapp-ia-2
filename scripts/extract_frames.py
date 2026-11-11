#!/usr/bin/env python3
"""Extract training frames from a match video for a YOLO dataset.

Saves one frame every N seconds into ``dataset/images/raw`` using consecutive names
(``frame_000001.jpg``...), keeping the original resolution. Near-duplicate frames are skipped
using a small grayscale thumbnail comparison, which keeps the dataset diverse without
comparing full-resolution images.

Only OpenCV and NumPy are required (both already included in this project).

Examples:
    python scripts/extract_frames.py --video videos/okliga_prueba.mp4
    python scripts/extract_frames.py --video match.mp4 --interval 0.5 --output dataset/images/raw
    python scripts/extract_frames.py --video match.mp4 --start 60 --end 300 --no-dedupe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OUTPUT = Path("dataset/images/raw")
THUMBNAIL_SIZE = (64, 36)
JPEG_QUALITY = 95


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae fotogramas de un vídeo para etiquetar un dataset de YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True, help="archivo de vídeo de entrada")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="segundos entre fotogramas extraídos (1, 2, 5...)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="carpeta donde guardar los JPG"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="segundo en el que empezar a extraer"
    )
    parser.add_argument(
        "--end", type=float, default=None, help="segundo en el que parar (por defecto, final)"
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.015,
        help="diferencia media mínima (0-1) para considerar dos fotogramas distintos",
    )
    parser.add_argument(
        "--no-dedupe", action="store_true", help="guardar todos los fotogramas, sin filtrar"
    )
    parser.add_argument("--prefix", default="frame", help="prefijo del nombre de archivo")
    return parser.parse_args(argv)


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    """Open the video and return (capture, fps, frame_count, width, height)."""
    if not path.exists():
        raise FileNotFoundError(f"no existe el vídeo: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV no puede abrir el vídeo: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1.0:
        raise RuntimeError(f"FPS no válidos en {path}: {fps}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return capture, float(fps), frame_count, width, height


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """Small grayscale version of a frame, used for the duplicate check."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)


def is_duplicate(candidate: np.ndarray, recent: list[np.ndarray], threshold: float) -> bool:
    """True when the thumbnail is too similar to any recently saved frame."""
    if not recent:
        return False
    for previous in recent:
        difference = float(np.mean(cv2.absdiff(candidate, previous))) / 255.0
        if difference < threshold:
            return True
    return False


def extract(
    video: Path,
    output: Path,
    interval: float,
    start: float,
    end: float | None,
    dedupe_threshold: float,
    dedupe: bool,
    prefix: str,
) -> dict[str, object]:
    """Walk the video and write the selected frames. Returns a summary dictionary."""
    capture, fps, frame_count, width, height = open_video(video)
    output.mkdir(parents=True, exist_ok=True)

    step = max(round(fps * interval), 1)
    first_frame = max(int(start * fps), 0)
    last_frame = int(end * fps) if end is not None else (frame_count or 2**31 - 1)

    recent_thumbnails: list[np.ndarray] = []
    saved = 0
    skipped_duplicates = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index > last_frame:
                break
            if index >= first_frame and (index - first_frame) % step == 0:
                thumb = thumbnail(frame)
                if dedupe and is_duplicate(thumb, recent_thumbnails, dedupe_threshold):
                    skipped_duplicates += 1
                else:
                    saved += 1
                    target = output / f"{prefix}_{saved:06d}.jpg"
                    ok_write = cv2.imwrite(
                        str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )
                    if not ok_write:
                        raise RuntimeError(f"no se pudo escribir {target}")
                    recent_thumbnails.append(thumb)
                    # Keep the memory footprint small: only the latest frames are compared.
                    if len(recent_thumbnails) > 30:
                        recent_thumbnails.pop(0)
            index += 1
    finally:
        capture.release()

    return {
        "duration_s": (frame_count / fps) if frame_count else index / fps,
        "fps": fps,
        "frame_count": frame_count or index,
        "resolution": f"{width}x{height}",
        "saved": saved,
        "skipped_duplicates": skipped_duplicates,
        "output": output.resolve(),
    }


def print_summary(summary: dict[str, object]) -> None:
    print("\n--- Resumen ---")
    print(f"Duración del vídeo      : {summary['duration_s']:.2f} s")
    print(f"FPS                     : {summary['fps']:.2f}")
    print(f"Fotogramas totales      : {summary['frame_count']}")
    print(f"Resolución              : {summary['resolution']}")
    print(f"Fotogramas guardados    : {summary['saved']}")
    print(f"Duplicados descartados  : {summary['skipped_duplicates']}")
    print(f"Carpeta de salida       : {summary['output']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.interval <= 0:
        print("error: --interval debe ser mayor que 0", file=sys.stderr)
        return 2
    try:
        summary = extract(
            video=args.video,
            output=args.output,
            interval=args.interval,
            start=max(args.start, 0.0),
            end=args.end,
            dedupe_threshold=args.dedupe_threshold,
            dedupe=not args.no_dedupe,
            prefix=args.prefix,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print_summary(summary)
    if summary["saved"] == 0:
        print("aviso: no se guardó ningún fotograma; revisa --start/--end y --interval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
