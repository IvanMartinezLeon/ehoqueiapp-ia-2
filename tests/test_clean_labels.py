"""Tests for the geometric label-cleanup rules."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clean_labels import (  # pyright: ignore[reportMissingImports]
    Box,
    CleanRules,
    clean_boxes,
    parse_boxes,
)

FRAME_W = 3840
FRAME_H = 2160


def _box(class_id: int, height_px: float, width_px: float = 100.0) -> str:
    return f"{class_id} 0.5 0.5 {width_px / FRAME_W:.6f} {height_px / FRAME_H:.6f}"


def _run(lines: list[str], rules: CleanRules | None = None) -> tuple[list[Box], dict[str, int]]:
    return clean_boxes(parse_boxes(lines), FRAME_W, FRAME_H, rules or CleanRules())


def test_parse_boxes_ignores_malformed_lines() -> None:
    boxes = parse_boxes([_box(0, 150), "0 0.5 0.5", "hola", ""])
    assert len(boxes) == 1
    assert boxes[0].class_id == 0


def test_small_boxes_are_removed() -> None:
    kept, removed = _run([_box(0, 30), _box(0, 200)])
    assert len(kept) == 1
    assert removed["small"] == 1


def test_large_boxes_are_removed() -> None:
    kept, removed = _run([_box(0, 700), _box(0, 200)])
    assert len(kept) == 1
    assert removed["large"] == 1


def test_extra_goalkeepers_keep_the_largest() -> None:
    kept, removed = _run([_box(1, 120, 40), _box(1, 200, 90), _box(1, 160, 60)])
    assert removed["extra_goalkeeper"] == 1
    assert len(kept) == 2
    assert all(box.class_id == 1 for box in kept)
    assert max(box.area() for box in kept) > min(box.area() for box in kept)


def test_goalkeeper_limit_can_be_disabled() -> None:
    rules = CleanRules(max_goalkeepers=0)
    kept, removed = _run([_box(1, 120, 40), _box(1, 200, 90), _box(1, 160, 60)], rules)
    assert len(kept) == 3
    assert removed["extra_goalkeeper"] == 0


def test_order_is_preserved_and_classes_never_change() -> None:
    lines = [_box(2, 200), _box(0, 300), _box(1, 250)]
    kept, removed = _run(lines)
    assert [box.line for box in kept] == lines
    assert sum(removed.values()) == 0


def test_custom_limits_are_honoured() -> None:
    rules = CleanRules(min_height_px=100, max_height_px=400)
    kept, removed = _run([_box(0, 90), _box(0, 200), _box(0, 500)], rules)
    assert len(kept) == 1
    assert removed == {"small": 1, "large": 1, "extra_goalkeeper": 0, "class": 0}


def test_dropped_classes_are_removed_entirely() -> None:
    rules = CleanRules(dropped_classes=(3,))
    kept, removed = _run([_box(0, 200), _box(3, 40), _box(3, 90)], rules)
    assert [box.class_id for box in kept] == [0]
    assert removed["class"] == 2


def test_dropped_classes_ignore_geometry() -> None:
    rules = CleanRules(dropped_classes=(3,))
    kept, removed = _run([_box(3, 900)], rules)
    assert kept == []
    assert removed["class"] == 1
    assert removed["large"] == 0
