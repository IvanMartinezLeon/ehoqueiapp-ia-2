"""Rendering of tracked boxes onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from .tracking import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
FALLBACK_COLOR = (200, 200, 200)

# Distinct BGR colours; a track id maps to a slot deterministically.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (60, 76, 231),
    (219, 152, 52),
    (34, 126, 230),
    (182, 89, 155),
    (152, 223, 86),
    (69, 190, 251),
    (129, 104, 233),
    (93, 177, 34),
    (222, 143, 92),
    (60, 90, 190),
    (34, 202, 156),
    (255, 160, 194),
)
DARK_LUMINANCE = 90


def color_for(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour for a track id (same id -> same colour every run)."""
    if track_id is None:
        return FALLBACK_COLOR
    return PALETTE[int(track_id) % len(PALETTE)]


def _text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """White text on dark boxes, black text on light ones."""
    blue, green, red = background
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (255, 255, 255) if luminance < DARK_LUMINANCE else (0, 0, 0)


def annotate(
    frame: np.ndarray,
    tracks: list[Track],
    show_conf: bool = True,
    colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with boxes, ids and optional confidence drawn.

    ``colors`` maps a track id to its box colour (used for team classification).
    Tracks missing from the map fall back to the per-id palette.
    """
    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (round(v) for v in track.bbox)
        color = (colors or {}).get(track.track_id, color_for(track.track_id))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, THICKNESS)
        label = f"#{track.track_id}"
        if show_conf:
            label += f" {track.conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        top = max(y1 - text_h - baseline - 2, 0)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 4, top + text_h + baseline + 2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, top + text_h + 1),
            FONT,
            FONT_SCALE,
            _text_color(color),
            1,
            cv2.LINE_AA,
        )
    return canvas
