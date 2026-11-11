#!/usr/bin/env python3
"""Interactive pitch calibration: click 4 floor points and give them real pitch coordinates."""

from __future__ import annotations

import argparse
import sys
from math import hypot
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from player_detect.pitch import Calibration, order_quad, save_calibration

WINDOW = (
    "calibracion - 4 puntos del suelo - clic: anadir | Z: deshacer | R: reiniciar "
    "| ENTER: guardar | ESC: salir"
)
MARKER_RADIUS = 5
MIN_CLICK_SEPARATION_PX = 8
HINT = "clic: anadir | Z: deshacer | R: reiniciar | ENTER: guardar | ESC: salir"


def _accepts_click(
    clicks: list[tuple[float, float]],
    x: float,
    y: float,
    count: int = 4,
    min_separation_px: float = MIN_CLICK_SEPARATION_PX,
) -> bool:
    """True when a click at ``(x, y)`` adds a new, distinct corner.

    Rejects extra clicks beyond ``count`` and duplicates (a double click fires two events at
    the same pixel, which used to make the calibration degenerate).
    """
    if len(clicks) >= count:
        return False
    return all(hypot(x - cx, y - cy) >= min_separation_px for cx, cy in clicks)


def _parse_points(raw: str) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for chunk in raw.split():
        try:
            x_text, y_text = chunk.split(",")
            points.append((float(x_text), float(y_text)))
        except ValueError as error:
            raise ValueError(
                f"punto mal formado: {chunk!r}. Formato: 'x1,y1 x2,y2 x3,y3 x4,y4'"
            ) from error
    if len(points) != 4:
        raise ValueError("se esperan exactamente 4 puntos: 'x1,y1 x2,y2 x3,y3 x4,y4'")
    return tuple(points)


def _read_frame(video: Path, index: int = 0):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"no se puede abrir el vídeo: {video}")
    if index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"no se puede leer el frame {index} de {video}")
    return frame


def _click_points(frame, count: int = 4) -> tuple[tuple[float, float], ...]:
    clicks: list[tuple[float, float]] = []

    def render() -> None:
        display = frame.copy()
        for index, (x, y) in enumerate(clicks, start=1):
            point = (round(x), round(y))
            cv2.drawMarker(display, point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                display,
                str(index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        if len(clicks) > 1:
            cv2.polylines(
                display,
                [np.array(clicks, dtype="int32")],
                len(clicks) == count,
                (0, 255, 0),
                2,
            )
        cv2.putText(display, HINT, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(
            display,
            f"puntos: {len(clicks)}/{count}",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow(WINDOW, display)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if _accepts_click(clicks, float(x), float(y), count):
            clicks.append((float(x), float(y)))
            render()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    render()
    print(
        "Clic en 4 puntos distintos del suelo (esquinas de la pista; el orden da igual).\n"
        "Z deshace el ultimo punto, R lo reinicia, ENTER guarda con los 4, ESC cancela."
    )
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("calibración cancelada")
        if key == 13 and len(clicks) == count:
            break
        if key in (ord("z"), ord("Z"), 8, 127):
            if clicks:
                clicks.pop()
                render()
        elif key in (ord("r"), ord("R")):
            clicks.clear()
            render()
    cv2.waitKey(300)
    cv2.destroyAllWindows()
    return tuple(clicks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibración de pista por 4 puntos.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0, help="frame de referencia (def. 0)")
    parser.add_argument(
        "--width", type=float, default=40.0, help="ancho real del rectángulo en metros"
    )
    parser.add_argument(
        "--height", type=float, default=20.0, help="alto real del rectángulo en metros"
    )
    parser.add_argument(
        "--points", default=None, help="alternativa sin GUI: 'x1,y1 x2,y2 x3,y3 x4,y4'"
    )
    parser.add_argument("--out", type=Path, default=None, help="YAML de salida")
    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        print("error: --width y --height deben ser positivos", file=sys.stderr)
        return 2

    try:
        frame = _read_frame(args.video, args.frame)
        raw_points = _parse_points(args.points) if args.points else _click_points(frame)
        image_points = order_quad(raw_points)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rink_points = (
        (0.0, 0.0),
        (args.width, 0.0),
        (args.width, args.height),
        (0.0, args.height),
    )
    try:
        calibration = Calibration(
            video=str(args.video),
            frame=args.frame,
            image_points=image_points,
            rink_points=rink_points,
            width_m=args.width,
            height_m=args.height,
        )
    except ValueError as error:
        print(f"error de calibración: {error}", file=sys.stderr)
        return 2
    out = args.out or Path("config") / f"calibration_{args.video.stem}.yaml"
    save_calibration(calibration, out)
    print(f"calibración guardada en {out}")
    print(f"error medio de reproyección: {calibration.reprojection_error():.3f} m")
    for image_point, rink_point in zip(image_points, rink_points, strict=True):
        print(f"  pixel {image_point} -> pista {rink_point} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
