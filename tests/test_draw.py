"""Tests for the drawing module."""

from __future__ import annotations

import numpy as np

from player_detect.draw import annotate, color_for
from player_detect.tracking import Track


def _frame() -> np.ndarray:
    return np.full((120, 160, 3), 40, dtype=np.uint8)


def _track(track_id: int = 1) -> Track:
    return Track(track_id=track_id, bbox=(20.0, 20.0, 100.0, 90.0), conf=0.87)


def test_color_for_is_deterministic() -> None:
    assert color_for(5) == color_for(5)
    assert all(0 <= channel <= 255 for channel in color_for(42))


def test_color_for_varies_across_ids() -> None:
    colors = {color_for(track_id) for track_id in range(12)}
    assert len(colors) > 1


def test_annotate_does_not_mutate_input() -> None:
    frame = _frame()
    original = frame.copy()

    annotate(frame, [_track()])

    assert np.array_equal(frame, original)


def test_annotate_paints_inside_bbox() -> None:
    frame = _frame()

    canvas = annotate(frame, [_track()])

    assert canvas.shape == frame.shape
    assert not np.array_equal(canvas, frame)


def test_annotate_without_tracks_returns_copy() -> None:
    frame = _frame()

    canvas = annotate(frame, [])

    assert canvas is not frame
    assert np.array_equal(canvas, frame)
