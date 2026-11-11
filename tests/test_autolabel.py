"""Tests for the pre-annotation helpers (class selection and YOLO line writing)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``scripts/`` is not a package: make it importable so the autolabel helpers get unit-tested.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from autolabel import boxes_to_yolo, parse_class_ids  # pyright: ignore[reportMissingImports]


def test_parse_class_ids_accepts_list_with_spaces() -> None:
    assert parse_class_ids("0, 1,2") == [0, 1, 2]


def test_parse_class_ids_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_class_ids(" , ")


def test_boxes_to_yolo_keeps_per_box_classes() -> None:
    boxes = [(0.0, 0.0, 100.0, 50.0, 0.9), (10.0, 10.0, 60.0, 50.0, 0.8)]
    lines = boxes_to_yolo(boxes, width=200, height=100, class_ids=[1, 2], min_height=10)
    assert [line.split()[0] for line in lines] == ["1", "2"]


def test_boxes_to_yolo_normalises_centre_and_size() -> None:
    lines = boxes_to_yolo(
        [(50.0, 25.0, 150.0, 75.0, 0.9)], width=200, height=100, class_ids=[0], min_height=10
    )
    assert lines == ["0 0.500000 0.500000 0.500000 0.500000"]


def test_boxes_to_yolo_drops_short_and_thin_boxes() -> None:
    boxes = [(0.0, 0.0, 100.0, 5.0, 0.9), (0.0, 0.0, 2.0, 80.0, 0.9)]
    assert boxes_to_yolo(boxes, 200, 100, [0, 0], min_height=10) == []


def test_boxes_to_yolo_requires_one_class_per_box() -> None:
    with pytest.raises(ValueError):
        boxes_to_yolo([(0.0, 0.0, 10.0, 10.0, 0.9)], 200, 100, [0, 1], min_height=1)
